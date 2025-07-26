import sys
import webbrowser
import ssl
import random
import warnings
import nltk
import spacy

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    
try:
    import language_tool_python
    LANGUAGETOOL_AVAILABLE = True
except ImportError:
    LANGUAGETOOL_AVAILABLE = False

from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QGridLayout, QPushButton, QTextEdit, QCheckBox, QSlider, 
                               QLabel, QFrame, QMessageBox, QComboBox)
from PySide6.QtCore import Qt, QObject, Signal, QRunnable, QThreadPool
from PySide6.QtGui import QFont

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

STRINGS = {
    "window_title": {"en": "Toprak AI Humanizer - Expert", "tr": "Toprak AI Humanizer - Uzman"},
    "main_title": {"en": "Toprak AI Humanizer", "tr": "Toprak AI Humanizer"},
    "options_frame_title": {"en": "Conversion Options", "tr": "Dönüşüm Seçenekleri"},
    "passive_option": {"en": "Passive Voice", "tr": "Pasif Cümle"},
    "synonyms_option": {"en": "Synonyms", "tr": "Eş Anlamlı"},
    "restructure_option": {"en": "Restructure", "tr": "Yapılandırma"},
    "paraphrase_option": {"en": "Deep Rewrite", "tr": "Derin Yazma"},
    "sentiment_option": {"en": "Sentiment Guard", "tr": "Duygu Koruması"},
    "grammar_option": {"en": "Grammar Correction", "tr": "Dilbilgisi Düzeltme"},
    "creativity_label": {"en": "Creativity:", "tr": "Yaratıcılık:"},
    "safe_level": {"en": "Safe", "tr": "Güvenli"},
    "balanced_level": {"en": "Balanced", "tr": "Dengeli"},
    "creative_level": {"en": "Creative", "tr": "Yaratıcı"},
    "input_label": {"en": "Input Text", "tr": "Giriş Metni"},
    "output_label": {"en": "Output Text", "tr": "Çıktı Metni"},
    "input_placeholder": {"en": "Paste the text to be humanized here...", "tr": "İnsanlaştırılacak metni buraya yapıştırın..."},
    "output_placeholder": {"en": "The result will appear here...", "tr": "Sonuç burada görünecek..."},
    "convert_button": {"en": "Convert", "tr": "Dönüştür"},
    "github_button": {"en": "GitHub", "tr": "GitHub"},
    "how_to_use_button": {"en": "How to Use", "tr": "Nasıl Kullanılır?"},
    "help_title": {"en": "How to Use Guide", "tr": "Nasıl Kullanılır Rehberi"},
    "help_disclaimer_title": {"en": "Warning", "tr": "Uyarı"},
    "help_disclaimer_text": {"en": "This program is an experimental tool. It may make incorrect or meaning-altering changes to the text. Please always check and verify the converted text carefully.", "tr": "Bu program deneysel bir çalışmadır. Metin üzerinde hatalı veya anlamı bozan değişiklikler yapabilir. Lütfen dönüştürülen metni her zaman dikkatlice kontrol edip doğrulayın."},
    "status_loading": {"en": "Loading models, please wait...", "tr": "Modeller yükleniyor, lütfen bekleyin..."},
    "status_ready": {"en": "Ready. You can convert the text.", "tr": "Hazır. Metni dönüştürebilirsiniz."},
    "status_converting": {"en": "Converting...", "tr": "Dönüştürülüyor..."},
    "status_done": {"en": "Conversion complete.", "tr": "Dönüşüm tamamlandı."},
    "status_error": {"en": "Error", "tr": "Hata"},
    "status_critical_error": {"en": "Critical Error: Models could not be loaded.", "tr": "Kritik Hata: Modeller yüklenemedi."},
    "warning_title": {"en": "Input Error", "tr": "Giriş Hatası"},
    "warning_text": {"en": "Please enter some text to convert.", "tr": "Lütfen dönüştürmek için bir metin girin."},
    "passive_tooltip": {"en": "This option attempts to convert active sentences into passive sentences. Its goal is to give the text a more formal or academic tone.", "tr": "Bu seçenek, aktif cümleleri pasif cümlelere dönüştürmeye çalışır. Amacı, metne daha resmi veya akademik bir ton katmaktır."},
    "synonyms_tooltip": {"en": "It replaces some words with their synonyms. The 'Creativity' slider determines how much the new word can deviate from the original's meaning.", "tr": "Kelimelerin bir kısmını, eş anlamlılarıyla değiştirir. 'Yaratıcılık' kaydırıcısı, yeni kelimenin orijinal anlamdan ne kadar uzaklaşabileceğini belirler."},
    "restructure_tooltip": {"en": "This option diversifies sentence structure by combining short sentences or splitting long ones. It helps break the monotonous pattern often found in AI-generated text.", "tr": "Bu seçenek, kısa cümleleri birleştirerek veya uzun cümleleri bölerek cümle yapısını çeşitlendirir. Yapay zeka metinlerinde sıkça bulunan monoton yapıyı kırmaya yardımcı olur."},
    "paraphrase_tooltip": {"en": "The most powerful feature. Uses a large AI model to completely rewrite sentences from scratch. This is very effective, but it is VERY SLOW.", "tr": "En güçlü özelliktir. Cümleleri sıfırdan tamamen yeniden yazmak için büyük bir yapay zeka modeli kullanır. Çok etkilidir, ancak ÇOK YAVAŞTIR."},
    "sentiment_tooltip": {"en": "Prevents word changes that reverse the meaning. For example, it stops a positive word like 'impact' from being replaced by a negative word like 'encroachment'.", "tr": "Kelime değişimlerinin, cümlenin anlamını tersine çevirmesini engeller. Örneğin, 'etki' gibi olumlu bir kelimenin 'istila' gibi olumsuz bir kelimeyle değiştirilmesini durdurur."},
    "grammar_tooltip": {"en": "Automatically corrects grammatical errors (e.g., subject-verb agreement) after the text has been modified. Requires Java to be installed on your system.", "tr": "Metin değiştirildikten sonra ortaya çıkan dilbilgisi hatalarını (örneğin özne-fiil uyumu) otomatik olarak düzeltir. Bilgisayarınızda Java'nın kurulu olmasını gerektirir."},
    "creativity_tooltip": {"en": "This slider controls the synonym selection and aggressiveness:\n- Safe: Makes fewer changes, prioritizing meaning.\n- Balanced: A good mix of variety and meaning preservation.\n- Creative: Makes more changes for variety, which might produce odd results.", "tr": "Bu kaydırıcı, eş anlamlı seçimini ve değiştirme sıklığını kontrol eder:\n- Güvenli: Anlamı korumak için daha az değişiklik yapar.\n- Dengeli: Anlam ve çeşitlilik arasında iyi bir denge kurar.\n- Yaratıcı: Çeşitlilik için daha fazla risk alır, bu da bazen garip sonuçlar üretebilir."},
}

DARK_STYLESHEET = """
QWidget { background-color: #2E3440; color: #ECEFF4; font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; }
QMainWindow { background-color: #2E3440; }
QTextEdit { background-color: #3B4252; border: 1px solid #4C566A; border-radius: 5px; padding: 8px; color: #D8DEE9; }
QPushButton { background-color: #5E81AC; color: #ECEFF4; border: none; padding: 10px 20px; border-radius: 5px; font-weight: bold; }
QPushButton:hover { background-color: #81A1C1; }
QPushButton:disabled { background-color: #4C566A; color: #D8DEE9; }
QPushButton#githubButton, QPushButton#helpButton { background-color: #434C5E; font-weight: normal; padding: 5px 15px; }
QPushButton#githubButton:hover, QPushButton#helpButton:hover { background-color: #4C566A; }
QCheckBox { spacing: 10px; }
QCheckBox::indicator { width: 18px; height: 18px; }
QCheckBox::indicator:unchecked { border: 1px solid #4C566A; background-color: #3B4252; border-radius: 4px; }
QCheckBox::indicator:checked { background-color: #88C0D0; border: 1px solid #88C0D0; border-radius: 4px; }
QSlider::groove:horizontal { border: 1px solid #4C566A; height: 8px; background: #3B4252; margin: 2px 0; border-radius: 4px; }
QSlider::handle:horizontal { background: #88C0D0; border: 1px solid #88C0D0; width: 18px; margin: -5px 0; border-radius: 9px; }
QLabel { color: #ECEFF4; }
QLabel#signatureLabel { color: #8F99AA; font-size: 12px; }
QFrame[objectName="optionsFrame"] { border: 1px solid #4C566A; border-radius: 5px; padding: 10px; }
QComboBox { border: 1px solid #4C566A; border-radius: 3px; padding: 5px; min-width: 6em; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 20px; border-left-width: 1px; border-left-color: #4C566A; border-left-style: solid; border-top-right-radius: 3px; border-bottom-right-radius: 3px; }
QComboBox QAbstractItemView { border: 1px solid #4C566A; background-color: #3B4252; color: #ECEFF4; selection-background-color: #5E81AC; }
QMessageBox { background-color: #FFFFFF; }
QMessageBox QLabel { color: #000000; background-color: #FFFFFF; }
"""

class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(tuple)
    progress = Signal(str)

class Runnable(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(self.signals.progress, *self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit((type(e), e, e.__traceback__))

def download_nltk_resources():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError: pass
    else: ssl._create_default_https_context = _create_unverified_https_context
    
    resources = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4']
    for resource in resources:
        try: nltk.data.find(f"tokenizers/{resource}")
        except LookupError: nltk.download(resource, quiet=True)

class Paraphraser:
    def __init__(self, progress_callback):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Paraphrasing requires 'transformers' and 'torch'.")
        model_name = "t5-base"
        progress_callback.emit(f"Loading paraphraser model '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        progress_callback.emit(f"Paraphraser model loaded on '{self.device}'.")

    def paraphrase(self, sentence, num_beams=5):
        text = "paraphrase: " + sentence
        encoding = self.tokenizer.encode_plus(text, return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_length=256, num_beams=num_beams, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

class AdvancedTextHumanizer:
    def __init__(self, progress_callback):
        progress_callback.emit("Checking NLTK resources...")
        download_nltk_resources()
        progress_callback.emit("NLTK resources are ready.")
        
        progress_callback.emit("Loading similarity model (MiniLM)...")
        self.nlp = spacy.load("en_core_web_sm")
        self.similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        progress_callback.emit("Similarity model loaded.")
        
        self.paraphraser = None
        self.grammar_tool = None
        self.academic_transitions = ["Moreover,", "Additionally,", "Furthermore,", "Hence,", "Therefore,", "Consequently,", "Nonetheless,", "Nevertheless,"]
        self.contraction_map = {"n't": " not", "'re": " are", "'s": " is", "'ll": " will", "'ve": " have", "'d": " would", "'m": " am"}

    def _initialize_paraphraser(self, progress_callback):
        if self.paraphraser is None: self.paraphraser = Paraphraser(progress_callback)
        return True
        
    def _initialize_grammar_tool(self, progress_callback):
        if self.grammar_tool is None:
            if not LANGUAGETOOL_AVAILABLE: raise ImportError("Grammar correction requires 'language-tool-python'.")
            progress_callback.emit("Loading grammar correction tool (requires Java)...")
            self.grammar_tool = language_tool_python.LanguageTool('en-US')
            progress_callback.emit("Grammar correction tool loaded.")
        return True

    def humanize_text(self, progress_callback, text, options):
        if options.get('use_paraphrase') and self.paraphraser is None: self._initialize_paraphraser(progress_callback)
        if options.get('use_grammar') and self.grammar_tool is None: self._initialize_grammar_tool(progress_callback)
        
        progress_callback.emit(STRINGS["status_converting"][options.get("lang", "en")])
        text = self.expand_contractions(text)
        
        if options.get('use_restructure'): text = self.restructure_sentences(text)
        
        doc = self.nlp(text)
        sents = list(doc.sents)
        transformed_sentences = []
        total_sents = len(sents)

        if total_sents == 0: return ""

        for i, sent in enumerate(sents):
            sentence_str = sent.text.strip()
            
            if options.get('use_paraphrase') and self.paraphraser and random.random() < 0.5:
                sentence_str = self.paraphraser.paraphrase(sentence_str)
            else:
                if options.get('use_passive') and random.random() < 0.3:
                    sentence_str = self.convert_to_passive(sentence_str)
                
                if options.get('use_synonyms'):
                    sentence_str = self.replace_with_synonyms(sentence_str, options)

                if random.random() < 0.3:
                    sentence_str = self.add_academic_transitions(sentence_str)

            if options.get('use_grammar') and self.grammar_tool:
                sentence_str = self.grammar_tool.correct(sentence_str)
            
            transformed_sentences.append(sentence_str)

        return ' '.join(transformed_sentences)

    def expand_contractions(self, text):
        for contraction, expansion in self.contraction_map.items(): text = text.replace(contraction, expansion)
        return text
        
    def add_academic_transitions(self, sentence):
        return f"{random.choice(self.academic_transitions)} {sentence}"

    def convert_to_passive(self, sentence):
        doc = self.nlp(sentence)
        for token in doc:
            if token.dep_ == 'dobj' and token.head.pos_ == 'VERB':
                verb = token.head
                subject = [t for t in verb.lefts if t.dep_ in ('nsubj', 'nsubjpass')]
                if subject: return f"{token.text.capitalize()} was {verb.lemma_} by {subject[0].text}."
        return sentence

    def restructure_sentences(self, text, p_restructure=0.4):
        doc = self.nlp(text)
        sents = list(doc.sents)
        new_sents_text = []
        i = 0
        while i < len(sents):
            if random.random() < p_restructure and i + 1 < len(sents) and len(sents[i]) < 12 and len(sents[i+1]) < 12:
                new_sents_text.append(sents[i].text.strip().rstrip('.') + ", and " + sents[i+1].text.strip().lower())
                i += 2
            else:
                new_sents_text.append(sents[i].text)
                i += 1
        return " ".join(new_sents_text)

    def replace_with_synonyms(self, sentence, options):
        tokens = word_tokenize(sentence)
        if len(tokens) > 50: return sentence
        
        creativity = options.get('creativity', 'safe')
        aggressiveness_map = {'safe': 0.25, 'balanced': 0.5, 'creative': 0.7}
        aggressiveness = aggressiveness_map.get(creativity, 0.5)

        pos_tags = nltk.pos_tag(tokens)
        new_tokens = list(tokens)
        for i, (word, pos) in enumerate(pos_tags):
            if pos.startswith(('J', 'N', 'V', 'R')) and len(word) > 3 and random.random() < aggressiveness:
                synonyms = self._get_synonyms(word, pos)
                if synonyms:
                    best_synonym = self._select_closest_synonym(word, synonyms, sentence, options)
                    if best_synonym: new_tokens[i] = best_synonym
        return ' '.join(new_tokens)

    def _get_synonyms(self, word, pos):
        wn_pos_map = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'R': wordnet.ADV, 'V': wordnet.VERB}
        wn_pos = wn_pos_map.get(pos[0])
        if not wn_pos: return []
        synonyms = set()
        for syn in wordnet.synsets(word, pos=wn_pos):
            for lemma in syn.lemmas():
                lemma_name = lemma.name().replace('_', ' ')
                if lemma_name.lower() != word.lower(): synonyms.add(lemma_name)
        return list(synonyms)

    def _select_closest_synonym(self, original_word, synonyms, context, options):
        if not synonyms: return None
        
        use_sentiment_guard = options.get('use_sentiment', False)
        creativity = options.get('creativity', 'safe')

        original_emb = self.similarity_model.encode(context)
        syn_scores = []
        for syn in synonyms:
            new_context = context.replace(original_word, syn)
            score = util.cos_sim(original_emb, self.similarity_model.encode(new_context))[0][0].item()
            syn_scores.append((syn, score))
        
        syn_scores.sort(key=lambda x: x[1], reverse=True)
        
        if use_sentiment_guard and TEXTBLOB_AVAILABLE:
            original_polarity = TextBlob(original_word).sentiment.polarity
            syn_scores = [ (s, score) for s, score in syn_scores if TextBlob(s).sentiment.polarity * original_polarity >= 0 ]

        threshold_map = {'safe': 0.9, 'balanced': 0.8, 'creative': 0.7}
        threshold = threshold_map.get(creativity, 0.9)
        valid_syns = [s for s, score in syn_scores if score > threshold]
        
        if not valid_syns: return None
        if creativity == 'safe': return valid_syns[0]
        else: return random.choice(valid_syns)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.current_lang = "en"
        self.humanizer = None
        self.threadpool = QThreadPool()

        self.init_ui()
        self.retranslate_ui()
        self.start_initial_loading()

    def get_string(self, key):
        return STRINGS.get(key, {}).get(self.current_lang, f"<{key}>")

    def init_ui(self):
        self.setStyleSheet(DARK_STYLESHEET)
        self.setGeometry(100, 100, 1400, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)

        top_bar_layout = QHBoxLayout()
        self.main_title_label = QLabel()
        title_font = QFont("Segoe UI", 28, QFont.Bold)
        self.main_title_label.setFont(title_font)
        self.main_title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        self.how_to_use_button = QPushButton()
        self.how_to_use_button.setObjectName("helpButton")
        self.how_to_use_button.clicked.connect(self.show_help_dialog)

        self.lang_combo = QComboBox()
        self.lang_combo.addItem("English", "en")
        self.lang_combo.addItem("Türkçe", "tr")
        self.lang_combo.currentIndexChanged.connect(self.on_language_change)

        top_bar_layout.addWidget(self.main_title_label)
        top_bar_layout.addStretch()
        top_bar_layout.addWidget(self.how_to_use_button)
        top_bar_layout.addWidget(self.lang_combo)
        main_layout.addLayout(top_bar_layout)

        self.options_frame = QFrame()
        self.options_frame.setObjectName("optionsFrame")
        main_layout.addWidget(self.options_frame)
        
        options_layout = QHBoxLayout(self.options_frame)
        
        self.options_vars = {
            'passive': QCheckBox(), 'synonyms': QCheckBox(), 'restructure': QCheckBox(), 
            'paraphrase': QCheckBox(), 'sentiment': QCheckBox(), 'grammar': QCheckBox()
        }
        self.options_vars['synonyms'].setChecked(True)
        self.options_vars['sentiment'].setChecked(True)

        for name, widget in self.options_vars.items():
            options_layout.addWidget(widget)

        if not TRANSFORMERS_AVAILABLE: self.options_vars['paraphrase'].setDisabled(True)
        if not TEXTBLOB_AVAILABLE: self.options_vars['sentiment'].setDisabled(True)
        if not LANGUAGETOOL_AVAILABLE: self.options_vars['grammar'].setDisabled(True)

        options_layout.addSpacing(20)

        slider_group_layout = QHBoxLayout()
        self.creativity_label_widget = QLabel()
        self.creativity_slider = QSlider(Qt.Horizontal)
        self.creativity_slider.setRange(0, 2)
        self.creativity_slider.setValue(1)
        self.creativity_slider.setFixedWidth(150)
        self.creativity_value_label = QLabel()
        self.creativity_value_label.setFixedWidth(70)
        self.creativity_slider.valueChanged.connect(self.update_creativity_label)
        slider_group_layout.addWidget(self.creativity_label_widget)
        slider_group_layout.addWidget(self.creativity_slider)
        slider_group_layout.addWidget(self.creativity_value_label)
        options_layout.addLayout(slider_group_layout)
        options_layout.addStretch()

        grid_layout = QGridLayout()
        main_layout.addLayout(grid_layout, 1)

        self.input_text_label = QLabel()
        self.output_text_label = QLabel()
        self.input_text = QTextEdit()
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)

        grid_layout.addWidget(self.input_text_label, 0, 0)
        grid_layout.addWidget(self.output_text_label, 0, 1)
        grid_layout.addWidget(self.input_text, 1, 0)
        grid_layout.addWidget(self.output_text, 1, 1)

        self.transform_button = QPushButton()
        self.transform_button.setDisabled(True)
        self.transform_button.clicked.connect(self.start_transformation)
        main_layout.addWidget(self.transform_button, alignment=Qt.AlignCenter)
        
        bottom_bar_layout = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignLeft)
        
        self.signature_label = QLabel("Made by. Toprak")
        self.signature_label.setObjectName("signatureLabel")

        self.github_button = QPushButton()
        self.github_button.setObjectName("githubButton")
        self.github_button.clicked.connect(self.open_github)

        bottom_bar_layout.addWidget(self.status_label)
        bottom_bar_layout.addStretch()
        bottom_bar_layout.addWidget(self.signature_label)
        bottom_bar_layout.addWidget(self.github_button)
        main_layout.addLayout(bottom_bar_layout)

    def retranslate_ui(self):
        self.setWindowTitle(self.get_string("window_title"))
        self.main_title_label.setText(self.get_string("main_title"))
        
        for name, widget in self.options_vars.items():
            widget.setText(self.get_string(f"{name}_option"))
        
        self.creativity_label_widget.setText(self.get_string("creativity_label"))
        self.update_creativity_label(self.creativity_slider.value())

        self.input_text_label.setText(self.get_string("input_label"))
        self.output_text_label.setText(self.get_string("output_label"))
        self.input_text.setPlaceholderText(self.get_string("input_placeholder"))
        self.output_text.setPlaceholderText(self.get_string("output_placeholder"))

        self.transform_button.setText(self.get_string("convert_button"))
        self.github_button.setText(self.get_string("github_button"))
        self.how_to_use_button.setText(self.get_string("how_to_use_button"))
        
        current_status = self.status_label.text()
        if self.transform_button.isEnabled():
            if "..." not in current_status: self.update_status("status_ready")
        else:
            if "..." not in current_status: self.update_status("status_loading")

    def show_help_dialog(self):
        title = self.get_string("help_title")
        full_text = f"""
        <p><b>{self.get_string("passive_option")}:</b><br>{self.get_string("passive_tooltip")}</p>
        <p><b>{self.get_string("synonyms_option")}:</b><br>{self.get_string("synonyms_tooltip")}</p>
        <p><b>{self.get_string("restructure_option")}:</b><br>{self.get_string("restructure_tooltip")}</p>
        <p><b>{self.get_string("paraphrase_option")}:</b><br>{self.get_string("paraphrase_tooltip")}</p>
        <p><b>{self.get_string("sentiment_option")}:</b><br>{self.get_string("sentiment_tooltip")}</p>
        <p><b>{self.get_string("grammar_option")}:</b><br>{self.get_string("grammar_tooltip")}</p>
        <hr>
        <p><b>{self.get_string("creativity_label")}</b><br>{self.get_string("creativity_tooltip")}</p>
        <hr>
        <p><b><u>{self.get_string("help_disclaimer_title")}:</u></b><br>{self.get_string("help_disclaimer_text")}</p>
        """
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(full_text)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec()

    def on_language_change(self, index):
        lang_code = self.lang_combo.itemData(index)
        if self.current_lang != lang_code:
            self.current_lang = lang_code
            self.retranslate_ui()

    def update_creativity_label(self, value):
        levels = {0: "safe_level", 1: "balanced_level", 2: "creative_level"}
        label_key = levels.get(value, "balanced_level")
        self.creativity_value_label.setText(self.get_string(label_key))

    def start_initial_loading(self):
        worker = Runnable(AdvancedTextHumanizer)
        worker.signals.finished.connect(self.on_loading_finished)
        worker.signals.error.connect(self.on_loading_error)
        worker.signals.progress.connect(self.update_status_direct)
        self.threadpool.start(worker)

    def on_loading_finished(self, humanizer_instance):
        self.humanizer = humanizer_instance
        self.transform_button.setDisabled(False)
        self.update_status("status_ready")

    def on_loading_error(self, error_tuple):
        error_msg = f"{self.get_string('status_critical_error')}: {error_tuple[1]}"
        self.update_status_direct(error_msg)
        QMessageBox.critical(self, self.get_string("status_error"), error_msg)
        
    def update_status_direct(self, message):
        self.status_label.setText(message)

    def update_status(self, key):
        self.status_label.setText(self.get_string(key))

    def open_github(self):
        webbrowser.open('https://github.com/toprak1224')

    def start_transformation(self):
        user_text = self.input_text.toPlainText().strip()
        if not user_text:
            QMessageBox.warning(self, self.get_string("warning_title"), self.get_string("warning_text"))
            return

        self.transform_button.setDisabled(True)
        self.output_text.clear()
        
        options = {f"use_{name}": widget.isChecked() for name, widget in self.options_vars.items()}
        creativity_map = {0: 'safe', 1: 'balanced', 2: 'creative'}
        options['creativity'] = creativity_map.get(self.creativity_slider.value(), 'balanced')
        options['lang'] = self.current_lang
        
        worker = Runnable(self.humanizer.humanize_text, user_text, options)
        worker.signals.finished.connect(self.on_transformation_finished)
        worker.signals.error.connect(self.on_transformation_error)
        worker.signals.progress.connect(self.update_status_direct)
        self.threadpool.start(worker)

    def on_transformation_finished(self, result):
        self.output_text.setPlainText(result)
        self.transform_button.setDisabled(False)
        self.update_status("status_done")
    
    def on_transformation_error(self, error_tuple):
        error_msg = f"{self.get_string('status_error')}: {error_tuple[1]}"
        self.update_status_direct(error_msg)
        QMessageBox.critical(self, self.get_string("status_error"), error_msg)
        self.transform_button.setDisabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())