"""
================================================================================
UTILS.PY - Funções Utilitárias para Análise de Texto e Machine Learning
================================================================================

Este arquivo contém todas as funções auxiliares utilizadas no projeto de 
classificação de textos com explicabilidade (LIME).

ESTRUTURA DO ARQUIVO:
1. Importações de Bibliotecas
2. Funções de Análise de Texto
3. Funções de Pré-processamento de Texto
4. Funções de Vetorização
5. Funções de Avaliação de Modelos
6. Funções de Treino de Modelos
7. Funções de Visualização e Análise
8. Funções de Similaridade

Autor: Miguel Maurício António
Última atualização: Dezembro 2025
================================================================================
"""

# ============================================================================
# 1. IMPORTAÇÕES DE BIBLIOTECAS
# ============================================================================

# Bibliotecas gerais
import pandas as pd
import numpy as np
import re
import string
from tqdm import tqdm
import matplotlib.pyplot as plt

# Bibliotecas de pré-processamento de texto
import emoji 
import nltk
from nltk.corpus import stopwords
import spacy 

# Bibliotecas de Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score as sk_accuracy, \
                            precision_score as sk_precision, \
                            recall_score as sk_recall, \
                            roc_auc_score as sk_roc_auc, \
                            classification_report
import seaborn as sns

# Bibliotecas de Interpretabilidade
from lime.lime_text import LimeTextExplainer



# ============================================================================
# 2. FUNÇÕES DE ANÁLISE DE TEXTO
# ============================================================================

def avgSizeWords(text):
    """
    Calcula o tamanho médio das palavras em um texto.
    
    Esta função divide o texto em palavras e calcula a média do número de 
    caracteres por palavra.
    
    Args:
        text (str): Texto a ser analisado
        
    Returns:
        float: Média do número de caracteres por palavra. Retorna 0 se o texto estiver vazio.
        
    Exemplo:
        >>> avgSizeWords("olá mundo teste")
        4.33  # (3 + 5 + 5) / 3
    """
    list_string = text.split()
    if not list_string:
        return 0
    chars = np.array([len(s) for s in list_string])
    return chars.mean()


def trucateText(text):
    """
    Trunca um texto para um máximo de 100 palavras.
    
    Esta função é útil para limitar o tamanho de textos muito longos,
    mantendo apenas as primeiras 100 palavras.
    
    Args:
        text (str): Texto a ser truncado
        
    Returns:
        str: Texto original (se tiver ≤100 palavras) ou texto truncado (primeiras 100 palavras)
        
    Exemplo:
        >>> trucateText("palavra " * 150)  # Texto com 150 palavras
        "palavra palavra ... palavra"  # Apenas 100 palavras
    """
    words = text.split()
    if len(words) <= 100:
        return text
    else:
        words = words[0:100]
        text = ' '.join(words)
        return text


# ============================================================================
# 3. FUNÇÕES DE PRÉ-PROCESSAMENTO DE TEXTO
# ============================================================================

# ----------------------------------------------------------------------------
# 3.1 Processamento de Emojis e Pontuação
# ----------------------------------------------------------------------------

# Listas de emojis e pontuação para processamento
emojis_list = list(emoji.EMOJI_DATA.keys())  # Lista de todos os emojis conhecidos
emojis_list += ['\n']  # Adiciona quebra de linha à lista
punct = list(string.punctuation) + ['\n']  # Lista de pontuação + quebra de linha
emojis_punct = emojis_list + punct  # Lista combinada


def processEmojisPunctuation(text, remove_punct=False, remove_emoji=False):
    """
    Processa emojis e pontuação em um texto.
    
    Esta função pode tanto remover quanto separar (adicionar espaços) emojis e pontuação.
    Separar é útil para que cada emoji/pontuação seja tratado como um token individual.
    
    Args:
        text (str): Texto a ser processado
        remove_punct (bool): Se True, remove pontuação. Se False, adiciona espaços ao redor
        remove_emoji (bool): Se True, remove emojis. Se False, adiciona espaços ao redor
        
    Returns:
        str: Texto processado com emojis/pontuação removidos ou separados
        
    Exemplo:
        >>> processEmojisPunctuation("Olá!😊Como vai?", remove_punct=False, remove_emoji=False)
        "Olá ! 😊 Como vai ?"
        >>> processEmojisPunctuation("Olá!😊Como vai?", remove_punct=True, remove_emoji=True)
        "Olá Como vai"
    """
    chars = set(text)
    for c in chars:
        # Processar pontuação
        if remove_punct:
            if c in punct:
                text = text.replace(c, ' ')  # Remove substituindo por espaço
        else:
            if c in punct:
                text = text.replace(c, ' ' + c + ' ')  # Separa com espaços

        # Processar emojis
        if remove_emoji:
            if c in emojis_list:
                text = text.replace(c, ' ')  # Remove substituindo por espaço
        else:
            if c in emojis_list:
                text = text.replace(c, ' ' + c + ' ')  # Separa com espaços

    # Remove espaços múltiplos
    text = re.sub(' +', ' ', text)
    return text


# ----------------------------------------------------------------------------
# 3.2 Remoção de Stopwords (Palavras Irrelevantes)
# ----------------------------------------------------------------------------

# Lista base de stopwords em português do NLTK
stop_words = list(stopwords.words('portuguese'))

# Stopwords adicionais específicas do domínio (redes sociais, abreviações, etc.)
new_stopwords = ['aí','pra','vão','vou','onde','lá','aqui',
                    'tá','pode','pois','so','deu','agora','todo',
                    'nao','ja','vc', 'bom', 'ai','ta', 'voce', 'alguem', 'ne', 'pq',
                    'cara','to','mim','la','vcs','tbm', 'tudo','mst', 'ip', 've', 
                    'td', 'msg', 'abs', 'ft', 
                    'rs', 'sqn', 'cmg', 
                    '03', '27', 
                    'http', 'https', 'www',
                    'tocantim']

# Combina as duas listas
stop_words = stop_words + new_stopwords

# Adiciona espaços ao redor de cada stopword para evitar remoções parciais
# Exemplo: ' de ' em vez de 'de' para não remover 'de' de 'desde'
final_stop_words = []
for sw in stop_words:
    sw = ' '+ sw + ' '
    final_stop_words.append(sw)


def removeStopwords(text):
    """
    Remove stopwords (palavras irrelevantes) de um texto.
    
    Stopwords são palavras muito comuns que geralmente não contribuem para o 
    significado do texto (ex: 'de', 'para', 'com', 'o', 'a', etc.).
    
    Args:
        text (str): Texto do qual remover stopwords
        
    Returns:
        str: Texto sem stopwords
        
    Exemplo:
        >>> removeStopwords(" eu vou para a casa ")
        " eu casa "  # 'vou', 'para', 'a' foram removidas
    """
    for sw in final_stop_words:
        text = text.replace(sw,' ')
    # Remove espaços múltiplos
    text = re.sub(' +',' ',text)
    return text


# ----------------------------------------------------------------------------
# 3.3 Lematização (Redução de Palavras à Forma Base)
# ----------------------------------------------------------------------------

# Carrega o modelo de linguagem em português do spaCy
nlp = spacy.load('pt_core_news_sm')


def lemmatization(text):
    """
    Aplica lematização ao texto.
    
    Lematização reduz palavras à sua forma base (lema). Por exemplo:
    - "correndo", "correu", "correr" → "correr"
    - "gatos" → "gato"
    
    Isso ajuda a reduzir a dimensionalidade e agrupar palavras relacionadas.
    
    Args:
        text (str): Texto a ser lematizado
        
    Returns:
        str: Texto com palavras na forma lematizada
        
    Exemplo:
        >>> lemmatization("Os gatos estavam correndo rapidamente")
        "o gato estar correr rapidamente"
    """
    doc = nlp(text)
    lemmatized_tokens = []
    for token in doc:
        # Mantém pontuação e espaços como estão
        if token.is_punct or token.is_space:
             lemmatized_tokens.append(token.text)
        else:
             # Substitui pela forma lematizada
             lemmatized_tokens.append(token.lemma_)
    return " ".join(lemmatized_tokens)


# ----------------------------------------------------------------------------
# 3.4 Processamento de URLs
# ----------------------------------------------------------------------------

# VERSÃO ANTIGA (comentada): Extraía apenas o domínio da URL
'''def domainUrl(text):
    if 'http' in text:
        re_url = '[^\s]*https*://[^\s]*'
        matches = re.findall(re_url, text, flags=re.IGNORECASE)
        for m in matches:
            domain = m.split('//')
            domain = domain[1].split('/')[0]
            text = re.sub(re_url, domain, text, 1)
        return text
    else:
        return text'''


def domainUrl(text):
    """
    Remove URLs de um texto.
    
    Esta função identifica e remove todas as URLs (http, https) do texto,
    substituindo-as por espaços.
    
    Args:
        text (str): Texto contendo URLs
        
    Returns:
        str: Texto sem URLs
        
    Exemplo:
        >>> domainUrl("Veja isso https://exemplo.com/artigo aqui")
        "Veja isso aqui"
    """
    if 'http' in text:
        re_url = '[^\s]*https*://[^\s]*'  # Regex para identificar URLs
        matches = re.findall(re_url, text, flags=re.IGNORECASE)
        for m in matches:
            text = text.replace(m, ' ')  # Remove a URL
        text = re.sub(' +', ' ', text).strip()  # Remove espaços múltiplos
        return text
    else:
        return text


# ----------------------------------------------------------------------------
# 3.5 Processamento de Expressões Específicas
# ----------------------------------------------------------------------------

def processLoL(text):
    """
    Normaliza expressões de riso em português (kkk, kkkk, etc.).
    
    Em português, 'kkk' é usado para expressar riso. Esta função normaliza
    todas as variações (kkk, kkkk, kkkkk, etc.) para apenas 'kkk'.
    
    Args:
        text (str): Texto contendo expressões de riso
        
    Returns:
        str: Texto com expressões de riso normalizadas
        
    Exemplo:
        >>> processLoL("Isso é engraçado kkkkkkk muito bom kkkk")
        "Isso é engraçado kkk muito bom kkk"
    """
    re_kkk = 'kkk*'  # Regex para capturar kkk com qualquer quantidade de k's
    t = re.sub(re_kkk, "kkk", text, flags=re.IGNORECASE)
    return t


def firstSentence(text):
    """
    Extrai a primeira frase de um texto.
    
    Divide o texto por pontuação de fim de frase (.; ! ? ou quebra de linha)
    e retorna a primeira frase encontrada.
    
    Args:
        text (str): Texto completo
        
    Returns:
        str: Primeira frase do texto
        
    Exemplo:
        >>> firstSentence("Primeira frase. Segunda frase! Terceira?")
        "Primeira frase"
    """
    list_s = re.split('; |\. |\! |\? |\n',text)
    for s in list_s:
        if s is not None:
            return s


# ----------------------------------------------------------------------------
# 3.6 Correção Manual de Palavras
# ----------------------------------------------------------------------------

# Dicionário de correções manuais para palavras escritas incorretamente
correction_map = {
    'olher': 'olhar',      # Erro comum de digitação
    'erraddad': 'errado'   # Erro comum de digitação
}


def manual_correction(text, mapping):
    """
    Aplica correções manuais de palavras mal escritas.
    
    Usa um dicionário de mapeamento para corrigir palavras específicas que
    são frequentemente escritas incorretamente no dataset.
    
    Args:
        text (str): Texto a ser corrigido
        mapping (dict): Dicionário {palavra_errada: palavra_correta}
        
    Returns:
        str: Texto com correções aplicadas
        
    Exemplo:
        >>> manual_correction("vou olher isso", {'olher': 'olhar'})
        "vou olhar isso"
    """
    for wrong, right in mapping.items():
        # \b garante que apenas palavras completas sejam substituídas
        text = re.sub(r'\b' + re.escape(wrong) + r'\b', right, text)
    return text


# ----------------------------------------------------------------------------
# 3.7 Função Principal de Pré-processamento
# ----------------------------------------------------------------------------

def preprocess(text, semi=False, rpunct=False, remoji=False, sentence=False):
    """
    Aplica todas as etapas de pré-processamento a um texto.
    
    Esta é a função principal que orquestra todas as etapas de pré-processamento
    na ordem correta. Pode ser configurada para aplicar diferentes níveis de
    processamento.
    
    Args:
        text (str): Texto a ser processado
        semi (bool): Se True, retorna após processamento parcial (sem stopwords/lematização)
        rpunct (bool): Se True, remove pontuação (senão apenas separa)
        remoji (bool): Se True, remove emojis (senão apenas separa)
        sentence (bool): Se True, processa apenas a primeira frase
        
    Returns:
        str: Texto pré-processado
        
    Pipeline de processamento:
        1. Extrai primeira frase (se sentence=True)
        2. Converte para minúsculas
        3. Aplica correções manuais
        4. Remove URLs
        5. Normaliza expressões de riso (kkk)
        6. Processa emojis e pontuação
        7. Remove stopwords (se semi=False)
        8. Aplica lematização (se semi=False)
        
    Exemplo:
        >>> preprocess("Olá! Como vai? https://site.com kkkkk 😊")
        "olá ir kkk 😊"  # (simplificado)
    """
    # 1. Extrai primeira frase se necessário
    if sentence:
        text = firstSentence(text)
    
    # 2. Normalização básica
    text = text.lower().strip()
    
    # 3. Correções manuais
    text = manual_correction(text, correction_map)
    
    # 4. Remove URLs
    text = domainUrl(text)
    
    # 5. Normaliza expressões de riso
    text = processLoL(text)
    
    # 6. Processa emojis e pontuação
    text = processEmojisPunctuation(text, remove_punct=rpunct, remove_emoji=remoji)
    
    # 7. Se semi=True, retorna aqui (processamento parcial)
    if semi:
        return text
    
    # 8. Remove stopwords
    text = removeStopwords(text)
    
    # 9. Aplica lematização
    text = lemmatization(text)
    
    return text


# ============================================================================
# 4. FUNÇÕES DE VETORIZAÇÃO
# ============================================================================

def defineVectorizing(experiment):
    """
    Define e configura o vetorizador apropriado baseado no nome do experimento.
    
    Esta função cria um vetorizador (Bag-of-Words ou TF-IDF) com configurações
    específicas de n-gramas baseado no nome do experimento.
    
    Args:
        experiment (str): Nome do experimento no formato 'vectorizer-ngram[-max_features]'
                         Exemplos: 'bow-unigram', 'tfidf-unigram_bigram-max_features'
        
    Returns:
        CountVectorizer ou TfidfVectorizer: Vetorizador configurado
        
    Configurações do experimento:
        - Vetorizador: 'bow' (Bag of Words) ou 'tfidf' (TF-IDF)
        - N-gramas: 'unigram' (1,1), 'unigram_bigram' (1,2), 'unigram_bigram_trigram' (1,3)
        - max_features: Se presente no nome, limita a 5000 features
        - min_df: N-gramas que aparecem menos de 5 vezes são ignorados
        
    Exemplo:
        >>> vec = defineVectorizing('tfidf-unigram_bigram-max_features')
        >>> # Retorna TfidfVectorizer com n-gramas (1,2) e max 5000 features
    """
    max_feat = None
    
    # Define o número máximo de features se especificado no experimento
    if 'max_features' in experiment:
        max_feat = 5000
    
    # Divide o nome do experimento em partes
    exp_parts = experiment.split('-')
    vec = exp_parts[0]  # Tipo de vetorizador (bow ou tfidf)
    ngram = exp_parts[1]  # Tipo de n-grama
    
    # Configura o range de n-gramas
    if ngram == 'unigram':
        ng = (1,1)  # Apenas palavras individuais
    elif ngram == 'unigram_bigram':
        ng = (1,2)  # Palavras individuais e pares de palavras
    elif ngram == 'unigram_bigram_trigram':
        ng = (1,3)  # Palavras individuais, pares e trios

    # Frequência mínima: n-gramas que aparecem menos de 5 vezes não são contados
    MIN_FREQUENCY = 5

    # Cria o vetorizador apropriado
    if vec == 'bow':
        # Bag of Words: conta presença/ausência de palavras (binary=True)
        vectorizer = CountVectorizer(
            max_features=max_feat,      # Limite de features (ou None)
            binary=True,                 # Apenas presença/ausência (não frequência)
            ngram_range=ng,              # Range de n-gramas
            lowercase=False,             # Não converte para minúsculas (já feito no preprocess)
            token_pattern=r'\b\w\w+\b',  # Padrão: palavras com 2+ caracteres
            min_df=MIN_FREQUENCY         # Frequência mínima
        )
    elif vec == 'tfidf':
        # TF-IDF: pondera pela frequência e raridade das palavras
        vectorizer = TfidfVectorizer(
            max_features=max_feat,       # Limite de features (ou None)
            ngram_range=ng,              # Range de n-gramas
            lowercase=False,             # Não converte para minúsculas (já feito no preprocess)
            token_pattern=r'\b\w\w+\b',  # Padrão: palavras com 2+ caracteres
            min_df=MIN_FREQUENCY         # Frequência mínima
        )

    return vectorizer


def vectorizing(vectorizer, texts_train, texts_test):
    """
    Aplica vetorização aos textos de treino e teste.
    
    Esta função treina o vetorizador no conjunto de treino e transforma
    tanto o treino quanto o teste em vetores numéricos.
    
    Args:
        vectorizer: Vetorizador já configurado (CountVectorizer ou TfidfVectorizer)
        texts_train (list): Lista de textos de treino (já pré-processados)
        texts_test (list): Lista de textos de teste (já pré-processados)
        
    Returns:
        tuple: (X_train, X_test) - Matrizes esparsas com os vetores de features
        
    Processo:
        1. Aprende o vocabulário do conjunto de treino
        2. Transforma textos de treino em vetores usando esse vocabulário
        3. Transforma textos de teste em vetores usando o mesmo vocabulário
        
    Nota:
        É crucial que o vetorizador seja treinado APENAS no conjunto de treino
        para evitar data leakage.
        
    Exemplo:
        >>> vec = defineVectorizing('tfidf-unigram')
        >>> X_train, X_test = vectorizing(vec, train_texts, test_texts)
        Train: (8000, 5000)  # 8000 amostras, 5000 features
        Test: (2000, 5000)   # 2000 amostras, mesmas 5000 features
    """
    # Aprende o vocabulário apenas do conjunto de treino
    vectorizer.fit(texts_train)
    
    # Transforma os textos em vetores usando o vocabulário aprendido
    X_train = vectorizer.transform(texts_train)
    X_test = vectorizer.transform(texts_test)
    
    # Imprime as dimensões para verificação
    print('Train:', X_train.shape)
    print('Test:', X_test.shape)
    
    return X_train, X_test


# ============================================================================
# 5. FUNÇÕES DE AVALIAÇÃO DE MODELOS
# ============================================================================

def getTestMetrics(y_true, y_pred, y_prob=None, full_metrics=False, class_names=None):
    """
    Calcula métricas de avaliação para um modelo de classificação.
    
    Esta função calcula diversas métricas de desempenho e gera um relatório
    de classificação completo.
    
    Args:
        y_true (array): Labels verdadeiros
        y_pred (array): Predições do modelo
        y_prob (array, opcional): Probabilidades preditas (para calcular AUC)
        full_metrics (bool): Se True, imprime todas as métricas
        class_names (list, opcional): Nomes das classes para o relatório
        
    Returns:
        tuple: (accuracy, precision, precision_neg, recall, recall_neg, 
                f1, f1_neg, roc_auc, report_str)
        
    Métricas calculadas:
        - Accuracy: Proporção de predições corretas
        - Precision (weighted): Precisão média ponderada por classe
        - Recall (weighted): Recall médio ponderado por classe
        - F1 (weighted): F1-score médio ponderado por classe
        - ROC-AUC: Área sob a curva ROC (se y_prob fornecido)
        - Classification Report: Relatório detalhado por classe
    """
    # Calcula métricas principais
    acc = sk_accuracy(y_true, y_pred)
    precision = sk_precision(y_true, y_pred, average='weighted')
    recall = sk_recall(y_true, y_pred, average='weighted')
    
    # Calcula F1 manualmente para evitar problemas de arredondamento
    epsilon = 1e-7  # Evita divisão por zero
    f1 = 2 * (precision * recall) / (precision + recall + epsilon) if (precision + recall) > 0 else 0

    # Tenta calcular ROC-AUC (pode falhar se y_prob não fornecido)
    try:
        roc_auc = sk_roc_auc(y_true, y_prob, multi_class='ovr')
    except Exception:
        roc_auc = np.nan

    # Métricas para classe negativa (não utilizadas atualmente)
    precision_neg = recall_neg = f1_neg = np.nan
    
    # Gera o relatório de classificação como string
    report_str = classification_report(y_true, y_pred, target_names=class_names, output_dict=False)
    
    # Imprime métricas se solicitado
    if full_metrics:
        print(f"## 📊 Métricas de Desempenho (Weighted) ##")
        print(f"Accuracy: {acc:.3f}")
        print(f"Precision (W): {precision:.3f}")
        print(f"Recall (W): {recall:.3f}")
        print(f"F1 (W): {f1:.3f}")
        print(f"AUC: {roc_auc:.3f}")
        print("\n---")
        print("## 📋 Classification Report ##")
        print(report_str)

    return acc, precision, precision_neg, recall, recall_neg, f1, f1_neg, roc_auc, report_str


def save_reports_to_txt(models_results, filename='classifications_reports.txt'):
    """
    Salva os relatórios de classificação de múltiplos modelos em um arquivo .txt.
    
    Esta função é útil para documentar e comparar os resultados de diferentes
    modelos em um formato legível.
    
    Args:
        models_results (dict): Dicionário onde:
                              - chave: nome do modelo (str)
                              - valor: tupla (modelo_treinado, metricas)
        filename (str): Nome do arquivo de saída (padrão: 'classifications_reports.txt')
        
    Formato do arquivo:
        RELATÓRIOS DE CLASSIFICAÇÃO DOS MODELOS
        ==================================================
        
        === Modelo: Logistic Regression ===
        [relatório de classificação]
        --------------------------------------------------
        
        === Modelo: Random Forest ===
        [relatório de classificação]
        --------------------------------------------------
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("RELATÓRIOS DE CLASSIFICAÇÃO DOS MODELOS\n")
        f.write("="*50 + "\n\n")
        
        for model_name, (model_obj, metrics) in models_results.items():
            # O report_str é o último item da tupla de métricas
            report_str = metrics[-1] 
            
            f.write(f"=== Modelo: {model_name} ===\n")
            f.write(report_str)
            f.write("\n" + "-"*50 + "\n\n")
            
    print(f"Todos os relatórios foram salvos em '{filename}'")


# ============================================================================
# 6. FUNÇÕES DE VISUALIZAÇÃO E ANÁLISE
# ============================================================================

# Importações adicionais para visualização (já importadas no topo, mas listadas aqui para clareza)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report


def plot_bias_analysis(models_results, X_test, y_test, class_labels=['Real', 'Fake']):
    """
    Cria um gráfico de análise de viés (bias) entre classes para múltiplos modelos.
    
    Esta função visualiza a discrepância no F1-score entre duas classes para cada
    modelo, ajudando a identificar quais modelos têm viés em favor de uma classe.
    
    Args:
        models_results (dict): Dicionário onde:
                              - chave: nome do modelo (str)
                              - valor: tupla (modelo_treinado, metricas)
        X_test: Features do conjunto de teste
        y_test: Labels verdadeiros do conjunto de teste
        class_labels (list): Nomes das duas classes (padrão: ['Real', 'Fake'])
        
    Returns:
        DataFrame: Tabela com F1-scores por classe e gap para cada modelo
        
    Visualização:
        - Eixo Y: Modelos (ordenados por menor gap)
        - Eixo X: F1-score
        - Pontos: F1-score de cada classe
        - Linha: Conecta os dois F1-scores, mostrando o gap
        - Texto: Valor da diferença (gap) entre as classes
        
    Interpretação:
        - Gap pequeno: Modelo balanceado entre as classes
        - Gap grande: Modelo com viés em favor de uma classe
        
    Exemplo:
        >>> df = plot_bias_analysis(models_results, X_test, y_test)
        # Exibe gráfico e retorna DataFrame com os dados
    """
    data = []
    
    # 1. Extrai F1-scores de cada modelo para cada classe
    for model_name, (model, _) in models_results.items():
        # Faz predições
        y_pred = model.predict(X_test)
        
        # Gera relatório de classificação como dicionário
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Extrai apenas as chaves das classes (ignora 'accuracy', 'macro avg', etc.)
        keys = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        
        # Extrai F1-scores das duas classes
        f1_c0 = report[keys[0]]['f1-score']
        f1_c1 = report[keys[1]]['f1-score']
        
        # Armazena os dados
        data.append({
            'Model': model_name.replace('\n', ' '),  # Remove quebras de linha do nome
            f'{class_labels[0]}': f1_c0,
            f'{class_labels[1]}': f1_c1,
            'Gap': abs(f1_c0 - f1_c1)  # Diferença absoluta entre as classes
        })
    
    # 2. Cria DataFrame com os dados
    df = pd.DataFrame(data)
    
    # 3. Ordena por gap (menor gap = modelo mais balanceado aparece em cima)
    df = df.sort_values('Gap', ascending=True)

    # 4. Cria o gráfico
    plt.figure(figsize=(10, 8))
    
    # Desenha linhas horizontais conectando os F1-scores das duas classes
    plt.hlines(y=df['Model'], xmin=df[class_labels[0]], xmax=df[class_labels[1]], 
               color='grey', alpha=0.4, linewidth=3)
    
    # Desenha pontos para cada classe
    plt.scatter(df[class_labels[0]], df['Model'], color='#1f77b4', alpha=1, s=100, label=class_labels[0])
    plt.scatter(df[class_labels[1]], df['Model'], color='#ff7f0e', alpha=1, s=100, label=class_labels[1])
    
    # 5. Adiciona texto mostrando o gap para cada modelo
    for _, row in df.iterrows():
        # Calcula ponto médio entre os dois F1-scores
        mid_point = (row[class_labels[0]] + row[class_labels[1]]) / 2
        
        # Adiciona texto com o valor do gap
        plt.text(x=mid_point, y=row['Model'], s=f"diff: {row['Gap']:.2f}", 
                 color='#333333', fontsize=9, ha='center', va='bottom', fontweight='bold')

    # 6. Configurações de estilo do gráfico
    plt.title('Bias Analysis: F1-Score Discrepancy Between Classes', fontsize=14, fontweight='bold')
    plt.xlabel('F1-Score', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.legend(title='Class', loc='lower right')
    plt.margins(y=0.1)  # Adiciona margem vertical para melhor visualização

    plt.tight_layout()
    plt.show()
    
    return df



# ============================================================================
# 7. FUNÇÕES DE TREINO E AVALIAÇÃO DE MODELOS
# ============================================================================

def lr_eval(X_train, y_train, X_test, y_test):
    """
    Treina e avalia um modelo de Regressão Logística.
    
    Regressão Logística é um modelo linear simples e interpretável, ideal como
    baseline. Funciona bem para problemas linearmente separáveis.
    
    Args:
        X_train: Features de treino (matriz esparsa ou densa)
        y_train: Labels de treino
        X_test: Features de teste
        y_test: Labels de teste
        
    Returns:
        tuple: (modelo_treinado, metricas)
        
    Características do modelo:
        - Linear e interpretável
        - Rápido para treinar
        - Bom baseline para comparação
    """
    print('=== Logistic Regression ===')
    logreg = LogisticRegression().fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)[:, 1]
    metrics = getTestMetrics(y_test, y_pred, y_prob, full_metrics=True)
    return logreg, metrics


def nb_eval(X_train, y_train, X_test, y_test, experiment):
    """
    Treina e avalia um modelo Naive Bayes.
    
    Escolhe automaticamente entre BernoulliNB (para Bag-of-Words) e 
    ComplementNB (para TF-IDF) baseado no tipo de vetorização.
    
    Args:
        X_train: Features de treino
        y_train: Labels de treino
        X_test: Features de teste
        y_test: Labels de teste
        experiment: Tupla/lista contendo informações do experimento (primeiro elemento indica vetorização)
        
    Returns:
        tuple: (modelo_treinado, metricas)
        
    Características:
        - BernoulliNB: Para features binárias (Bag-of-Words)
        - ComplementNB: Para features contínuas (TF-IDF), lida melhor com desbalanceamento
        - Muito rápido para treinar
        - Assume independência entre features (simplificação forte)
    """
    if 'bow' in experiment[0]:
        print('=== Bernoulli Naive-Bayes ===')
        nb = BernoulliNB().fit(X_train, y_train)
    elif 'tfidf' in experiment[0]:
        print('=== Complement Naive-Bayes ===')
        nb = ComplementNB().fit(X_train, y_train)
    else:
        nb = BernoulliNB().fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    y_prob = nb.predict_proba(X_test)[:, 1]
    metrics = getTestMetrics(y_test, y_pred, y_prob, full_metrics=True)
    return nb, metrics


def lsvm_eval(X_train, y_train, X_test, y_test):
    """
    Treina e avalia uma SVM Linear (Support Vector Machine).
    
    SVM Linear encontra o hiperplano que melhor separa as classes com a
    maior margem possível.
    
    Args:
        X_train: Features de treino
        y_train: Labels de treino
        X_test: Features de teste
        y_test: Labels de teste
        
    Returns:
        tuple: (modelo_treinado, metricas)
        
    Características:
        - Eficiente para alta dimensionalidade (muitas features)
        - dual=False: Usa formulação primal (mais rápido para muitas features)
        - Não fornece probabilidades diretamente
        - Robusto e geralmente tem bom desempenho em texto
    """
    print('=== Linear Support Vector Machine ===')
    svm = LinearSVC(dual=False).fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    metrics = getTestMetrics(y_test, y_pred, full_metrics=True)
    return svm, metrics


def sgd_eval(X_train, y_train, X_test, y_test):
    """
    Treina e avalia uma SVM Linear com treinamento SGD (Stochastic Gradient Descent).
    
    Similar à LinearSVC, mas usa otimização por gradiente descendente estocástico,
    o que pode ser mais rápido para datasets muito grandes.
    
    Args:
        X_train: Features de treino
        y_train: Labels de treino
        X_test: Features de teste
        y_test: Labels de teste
        
    Returns:
        tuple: (modelo_treinado, metricas)
        
    Características:
        - Escalável para datasets grandes
        - Treinamento incremental (pode processar dados em batches)
        - Convergência pode ser menos estável que LinearSVC
    """
    print('=== Linear SVM with SGD training ===')
    sgd = SGDClassifier().fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    metrics = getTestMetrics(y_test, y_pred, full_metrics=True)
    return sgd, metrics


def svm_eval(X_train, y_train, X_test, y_test):
    """
    Treina e avalia uma SVM com kernel RBF (Radial Basis Function).
    
    SVM com kernel RBF pode capturar relações não-lineares entre features,
    mas é computacionalmente mais custosa.
    
    Args:
        X_train: Features de treino
        y_train: Labels de treino
        X_test: Features de teste
        y_test: Labels de teste
        
    Returns:
        tuple: (modelo_treinado, metricas)
        
    Características:
        - Pode capturar padrões não-lineares
        - probability=True: Habilita estimação de probabilidades
        - Mais lento que SVM linear
        - Pode ter overfitting se não bem regularizado
    """
    print('=== SVM with RBF kernel ===')
    svc = SVC(probability=True).fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    y_prob = svc.predict_proba(X_test)[:, 1]
    metrics = getTestMetrics(y_test, y_pred, y_prob, full_metrics=True)
    return svc, metrics


def knn_eval(X_train, y_train, X_test, y_test):
    """
    Treina e avalia um classificador K-Nearest Neighbors (KNN).
    
    KNN classifica baseado nas k amostras de treino mais próximas.
    É um método não-paramétrico e lazy (não treina, apenas memoriza).
    
    Args:
        X_train: Features de treino
        y_train: Labels de treino
        X_test: Features de teste
        y_test: Labels de teste
        
    Returns:
        tuple: (modelo_treinado, metricas)
        
    Características:
        - weights='distance': Vizinhos mais próximos têm mais peso
        - n_jobs=-1: Usa todos os cores disponíveis
        - Lento para predição em datasets grandes
        - Sensível à escala das features e à dimensionalidade
    """
    print('=== KNN ===')
    knn = KNeighborsClassifier(weights='distance', n_jobs=-1).fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)[:, 1]
    metrics = getTestMetrics(y_test, y_pred, y_prob, full_metrics=True)
    return knn, metrics


def rf_eval(X_train, y_train, X_test, y_test):
    """
    Treina e avalia um classificador Random Forest.
    
    Random Forest é um ensemble de árvores de decisão que vota para
    fazer a predição final. Robusto e geralmente tem bom desempenho.
    
    Args:
        X_train: Features de treino
        y_train: Labels de treino
        X_test: Features de teste
        y_test: Labels de teste
        
    Returns:
        tuple: (modelo_treinado, metricas)
        
    Características:
        - Ensemble de árvores de decisão
        - n_jobs=-1: Paralelização para treino mais rápido
        - Robusto a overfitting (comparado a uma única árvore)
        - Pode capturar interações não-lineares
        - Fornece importância de features
    """
    print('=== Random Forest ===')
    rf = RandomForestClassifier(n_jobs=-1).fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    metrics = getTestMetrics(y_test, y_pred, y_prob, full_metrics=True)
    return rf, metrics


def gb_eval(X_train, y_train, X_test, y_test):
    """
    Treina e avalia um classificador Gradient Boosting.
    
    Gradient Boosting constrói árvores sequencialmente, onde cada nova
    árvore corrige os erros das anteriores.
    
    Args:
        X_train: Features de treino
        y_train: Labels de treino
        X_test: Features de teste
        y_test: Labels de teste
        
    Returns:
        tuple: (modelo_treinado, metricas)
        
    Características:
        - n_estimators=200: Usa 200 árvores
        - Geralmente melhor desempenho que Random Forest
        - Mais propenso a overfitting que Random Forest
        - Treinamento sequencial (não paralelizável)
        - Mais lento para treinar
    """
    print('=== Gradient Boosting ===')
    gb = GradientBoostingClassifier(n_estimators=200).fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    y_prob = gb.predict_proba(X_test)[:, 1]
    metrics = getTestMetrics(y_test, y_pred, y_prob, full_metrics=True)
    return gb, metrics


def mlp_eval(X_train, y_train, X_test, y_test):
    """
    Treina e avalia um Multilayer Perceptron (Rede Neural).
    
    MLP é uma rede neural feedforward que pode aprender representações
    complexas e não-lineares dos dados.
    
    Args:
        X_train: Features de treino
        y_train: Labels de treino
        X_test: Features de teste
        y_test: Labels de teste
        
    Returns:
        tuple: (modelo_treinado, metricas)
        
    Características:
        - verbose=True: Mostra progresso do treinamento
        - early_stopping=True: Para quando validação não melhora
        - batch_size=64: Processa 64 amostras por vez
        - n_iter_no_change=5: Para após 5 épocas sem melhoria
        - tol=1e-3: Tolerância para critério de parada
        - Pode capturar padrões muito complexos
        - Requer mais dados e tuning que modelos mais simples
    """
    print('=== Multilayer Perceptron ===')
    mlp = MLPClassifier(
        verbose=True, early_stopping=True,
        batch_size=64, n_iter_no_change=5, tol=1e-3
    ).fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    y_prob = mlp.predict_proba(X_test)[:, 1]
    metrics = getTestMetrics(y_test, y_pred, y_prob, full_metrics=True)
    return mlp, metrics


def model_eval(model, X_train, y_train, X_test, y_test, experiment=None):
    """
    Função dispatcher que chama a função de avaliação apropriada para cada modelo.
    
    Esta função facilita o treinamento de diferentes modelos usando uma interface
    unificada, selecionando automaticamente a função correta baseada no nome do modelo.
    
    Args:
        model (str): Código do modelo a treinar. Opções:
                    - 'lr': Logistic Regression
                    - 'nb': Naive Bayes
                    - 'lsvm': Linear SVM
                    - 'sgd': SGD Classifier
                    - 'svm': SVM com kernel RBF
                    - 'knn': K-Nearest Neighbors
                    - 'rf': Random Forest
                    - 'gb': Gradient Boosting
                    - 'mlp': Multilayer Perceptron
        X_train: Features de treino
        y_train: Labels de treino
        X_test: Features de teste
        y_test: Labels de teste
        experiment (opcional): Informações do experimento (necessário para Naive Bayes)
        
    Returns:
        tuple: (modelo_treinado, metricas)
        
    Raises:
        ValueError: Se o código do modelo não for reconhecido
        
    Exemplo:
        >>> model, metrics = model_eval('rf', X_train, y_train, X_test, y_test)
        === Random Forest ===
        ## 📊 Métricas de Desempenho (Weighted) ##
        ...
    """
    if model == 'lr':
        return lr_eval(X_train, y_train, X_test, y_test)
    elif model == 'nb':
        return nb_eval(X_train, y_train, X_test, y_test, experiment)
    elif model == 'lsvm':
        return lsvm_eval(X_train, y_train, X_test, y_test)
    elif model == 'sgd':
        return sgd_eval(X_train, y_train, X_test, y_test)
    elif model == 'svm':
        return svm_eval(X_train, y_train, X_test, y_test)
    elif model == 'knn':
        return knn_eval(X_train, y_train, X_test, y_test)
    elif model == 'rf':
        return rf_eval(X_train, y_train, X_test, y_test)
    elif model == 'gb':
        return gb_eval(X_train, y_train, X_test, y_test)
    elif model == 'mlp':
        return mlp_eval(X_train, y_train, X_test, y_test)
    else:
        raise ValueError(f"Model '{model}' unknown.")
    


# ============================================================================
# 8. FUNÇÕES DE SIMILARIDADE E COMPARAÇÃO
# ============================================================================

def calculate_jaccard(set_a, set_b):
    """
    Calcula o índice de similaridade de Jaccard entre dois conjuntos.
    
    O índice de Jaccard mede a similaridade entre dois conjuntos calculando
    a razão entre a interseção e a união dos conjuntos.
    
    Args:
        set_a: Primeiro conjunto (ou lista que será convertida em conjunto)
        set_b: Segundo conjunto (ou lista que será convertida em conjunto)
        
    Returns:
        float: Índice de Jaccard entre 0.0 (completamente diferentes) e 
               1.0 (idênticos)
               
    Fórmula:
        Jaccard = |A ∩ B| / |A ∪ B|
        
    Interpretação:
        - 0.0: Conjuntos completamente disjuntos (sem elementos em comum)
        - 0.5: 50% de similaridade
        - 1.0: Conjuntos idênticos
        
    Exemplo:
        >>> calculate_jaccard([1, 2, 3], [2, 3, 4])
        0.5  # Interseção: {2, 3}, União: {1, 2, 3, 4}
        >>> calculate_jaccard(['a', 'b'], ['a', 'b'])
        1.0  # Conjuntos idênticos
    """
    # Converte para set se necessário
    if not isinstance(set_a, set): 
        set_a = set(set_a)
    if not isinstance(set_b, set): 
        set_b = set(set_b)
    
    # Calcula tamanho da interseção e união
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    
    # Evita divisão por zero
    if union == 0:
        return 0.0
    
    return intersection / union


def plot_models_metrics(models_results, save_path):
    """
    Cria um gráfico de barras comparando métricas de múltiplos modelos.
    
    Esta função gera uma visualização lado a lado das principais métricas
    (Accuracy, Precision, Recall, F1-Score) para todos os modelos treinados.
    
    Args:
        models_results (dict): Dicionário onde:
                              - chave: nome do modelo (str)
                              - valor: tupla (modelo_treinado, metricas)
        save_path (str): Caminho para salvar o gráfico (padrão: 'model_comparison.png')
                        Se None, não salva o gráfico
        
    Returns:
        tuple: (fig, ax, df) - Figura matplotlib, eixos e DataFrame com os dados
        
    Visualização:
        - Eixo X: Modelos
        - Eixo Y: Score (0.0 a 1.0)
        - Barras agrupadas: Uma para cada métrica
        - Cores: Paleta de verdes para as diferentes métricas
        
    Exemplo:
        >>> fig, ax, df = plot_models_metrics(models_results)
        Plot saved as 'model_comparison.png'
        
        Metrics table:
                    Model  Accuracy  Precision  Recall  F1-Score
        Logistic Regression     0.85       0.84    0.85      0.84
               Random Forest     0.88       0.87    0.88      0.87
        ...
    """
    # 1. Extrai dados das métricas de cada modelo
    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for name, (model, metrics) in models_results.items():
        model_names.append(name)
        accuracies.append(metrics[0])   # accuracy
        precisions.append(metrics[1])   # precision
        recalls.append(metrics[3])      # recall
        f1_scores.append(metrics[5])    # f1
    
    # 2. Cria DataFrame com os dados
    df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores
    })
    
    # 3. Configurações do gráfico
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Define posições das barras
    x = np.arange(len(df['Model']))
    width = 0.2  # Largura de cada barra
    
    # Paleta de cores (tons de verde)
    colors = ['#90EE90', '#66CDAA', '#3CB371', '#2E8B57']
    
    # 4. Cria as barras agrupadas para cada métrica
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        # Calcula offset para posicionar barras lado a lado
        offset = width * (i - 1.5)
        ax.bar(x + offset, df[metric], width, label=metric, color=color, 
               edgecolor='black', linewidth=0.7, alpha=0.9)
    
    # 5. Configurações de estilo
    ax.set_xlabel('Models', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], fontsize=10)
    ax.set_ylim(0.6, 0.69)  # Ajuste conforme necessário para seus dados
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)  # Grid atrás das barras
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9, ncol=4)
    
    plt.tight_layout()
    
    # 6. Salva o gráfico se caminho fornecido
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{save_path}'")
    
    plt.show()
    
    # 7. Imprime tabela de métricas
    print("\nMetrics table:")
    print(df.to_string(index=False))
    
    return fig, ax, df






import torch
import numpy as np
import pandas as pd
import Levenshtein
from bert_score import score as bert_score
from evaluate import load
from tqdm import tqdm

# ==========================================
# FUNÇÕES AUXILIARES DE MÉTRICAS
# ==========================================

def jaccard_similarity(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    if not a and not b: return 1.0
    return len(a & b) / len(a | b)

def calculate_perplexity(texts, model_id='gpt2'):
    """Calcula a fluidez do texto. Requer modelo de linguagem externo."""
    perplexity_metric = load("perplexity", module_type="metric")
    results = perplexity_metric.compute(model_id=model_id, add_start_token=False, predictions=texts)
    return results['mean_perplexity']

# ==========================================
# PIPELINE DE AVALIAÇÃO DE XAI
# ==========================================

def evaluate_xai_counterfactuals(df_eval, ptt5_model, ptt5_tokenizer, classifier, vectorizer):
    """
    df_eval: DataFrame com ['original_text', 'target_code']
    classifier: O seu modelo original (Random Forest/MLP)
    vectorizer: O TF-IDF vectorizer usado no treino do classificador
    """
    results = []
    generated_texts = []
    
    print("A gerar contrafactuais e a calcular métricas...")
    
    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval)):
        orig_text = row['original_text']
        code = row['target_code'] # ex: [negation]
        
        # 1. GERAR CONTRAFACTUAL COM PTT5
        input_text = f"gerar contrafactual {code}: {orig_text}"
        inputs = ptt5_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128).to(ptt5_model.device)
        
        with torch.no_grad():
            outputs = ptt5_model.generate(inputs.input_ids, max_length=128, num_beams=5)
        gen_text = ptt5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(gen_text)

        # 2. MÉTRICAS LINGUÍSTICAS
        dist = Levenshtein.distance(orig_text, gen_text)
        jaccard = jaccard_similarity(orig_text, gen_text)
        
        # 3. MÉTRICAS DE CLASSIFICAÇÃO (FLIP RATE & PROB SHIFT)
        # Transformar textos para o formato do classificador (TF-IDF)
        orig_vec = vectorizer.transform([orig_text])
        gen_vec = vectorizer.transform([gen_text])
        
        # Predições e Probabilidades
        orig_prob = classifier.predict_proba(orig_vec)[0]
        gen_prob = classifier.predict_proba(gen_vec)[0]
        
        orig_label = np.argmax(orig_prob)
        gen_label = np.argmax(gen_prob)
        
        # Flip ocorreu se a classe mudou
        flip = 1 if orig_label != gen_label else 0
        
        # Shift: Diferença na confiança da classe original
        prob_shift = orig_prob[orig_label] - gen_prob[orig_label]
        
        results.append({
            'orig': orig_text,
            'gen': gen_text,
            'flip': flip,
            'prob_shift': prob_shift,
            'levenshtein': dist,
            'jaccard': jaccard
        })

    # 4. MÉTRICAS AGREGADAS (BERTScore e Perplexity)
    print("A calcular BERTScore...")
    P, R, F1 = bert_score(generated_texts, [r['orig'] for r in results], lang="pt", verbose=False)
    
    print("A calcular Perplexity...")
    # Nota: gpt2 é base, para PT idealmente seria um modelo PT, mas gpt2 serve de proxy
    avg_ppl = calculate_perplexity(generated_texts) 

    # FINALIZAR RESULTADOS
    df_res = pd.DataFrame(results)
    
    metrics_summary = {
        "Flip Rate": df_res['flip'].mean(),
        "Avg Prob Shift": df_res['prob_shift'].mean(),
        "Avg Levenshtein": df_res['levenshtein'].mean(),
        "Avg Jaccard": df_res['jaccard'].mean(),
        "BERTScore F1": F1.mean().item(),
        "Perplexity": avg_ppl
    }
    
    return metrics_summary, df_res


# ============================================================================
# FIM DO ARQUIVO UTILS.PY
# ============================================================================