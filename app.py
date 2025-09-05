import os
import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.set_page_config(
    page_title="Dynamic Stats",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .tab-content {
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cabeçalho principal
st.markdown("""
<div class="main-header">
    <h1>🚀 Dynamic Stats</h1>
    <p style="font-size: 1.2rem; margin: 0;">Análise Estatística Dinâmica e Interativa</p>
    <p style="font-size: 1rem; margin: 0.5rem 0 0 0;">Dev: Dr Fernando Freua | Writed with Python, Powered by PyGWalker</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Utils (avisar Dr Fernando Freua se modificar) --------------------
@st.cache_data
def read_file(uploaded_file, sheet_name=None):
    if uploaded_file is None:
        return None
        
    name = uploaded_file.name.lower()
    try:
        if name.endswith((".xls", ".xlsx", ".xlsm")):
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
            # Se sheet_name não foi especificado e há múltiplas planilhas, pegar a primeira
            if isinstance(df, dict):
                if df:
                    # Pegar a primeira planilha disponível
                    first_sheet = list(df.keys())[0]
                    df = df[first_sheet]
                    st.info(f"📋 Múltiplas planilhas detectadas. Usando: {first_sheet}")
                else:
                    st.error("❌ Nenhuma planilha encontrada no arquivo Excel")
                    return None
        else:
            # Tentar detectar separador automaticamente
            data = uploaded_file.read()
            sample = data[:10000].decode("utf-8", errors="ignore")
            sep = ","
            if sample.count(";") > sample.count(","):
                sep = ";"
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=sep)
        
        # Verificar se o DataFrame foi criado corretamente
        if isinstance(df, pd.DataFrame):
            if not df.empty:
                return df
            else:
                st.error("❌ O arquivo carregado está vazio")
                return None
        else:
            st.error(f"❌ Tipo de dados inesperado: {type(df)}. Esperado: DataFrame")
            return None
            
    except Exception as e:
        st.error(f"❌ Falha ao ler o arquivo: {str(e)}")
        return None

@st.cache_data
def example_df():
    # Pequeno dataset de exemplo (Iris-like simplificado)
    try:
        csv = io.StringIO(
            """sepal_length,sepal_width,petal_length,petal_width,species\n"""
            + "\n".join([
                "5.1,3.5,1.4,0.2,azul",
                "4.9,3.0,1.4,0.2,azul",
                "6.2,3.4,5.4,2.3,vermelho",
                "6.9,3.1,5.1,2.3,vermelho",
                "5.9,3.0,4.2,1.5,preto",
                "6.0,2.2,4.0,1.0,preto",
                "5.5,2.3,4.0,1.3,preto",
                "5.0,3.6,1.4,0.2,azul",
                "6.5,3.0,5.2,2.0,amarelo",
                "5.7,2.8,4.1,1.3,verde",
            ])
        )
        df = pd.read_csv(csv)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
        else:
            st.error("❌ Erro ao criar dataset de exemplo")
            return None
    except Exception as e:
        st.error(f"❌ Erro ao criar dataset de exemplo: {str(e)}")
        return None

# Efeito de tamanho e helpers
def cohen_d(x, y):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    s = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    if s == 0:
        return np.nan
    return (x.mean() - y.mean()) / s

def cramers_v(chi2, n, r, c):
    return np.sqrt(chi2 / (n * (min(r-1, c-1)))) if min(r-1, c-1) > 0 else np.nan

# -------------------- Sidebar --------------------
st.sidebar.markdown('<div class="sidebar-header"><h3>📁 Gerenciar Dados</h3></div>', unsafe_allow_html=True)

use_example = st.sidebar.toggle("🎯 Carregar dataset de exemplo", value=False)

uploaded = st.sidebar.file_uploader("📤 Envie um CSV ou Excel", type=["csv","xls","xlsx","xlsm"])

# Botão de Reset
if st.sidebar.button("🔄 Reset Completo", type="secondary"):
    st.rerun()

# Botão Sobre
st.sidebar.markdown("---")
if st.sidebar.button("ℹ️ Sobre o Dynamic Stats"):
    st.sidebar.info("""
    **Dynamic Stats v2.0** 🚀
    
    Desenvolvido por Dr Fernando Freua
    
    Uma ferramenta completa para análise estatística dinâmica e exploração de dados interativa.
    
    **Recursos:**
    • 📊 Análise exploratória avançada
    • 🔬 Testes estatísticos robustos
    • 📈 Visualizações interativas
    • 🎨 Interface moderna e intuitiva
    """)

df = None
try:
    if use_example:
        df = example_df()
    elif uploaded is not None:
        df = read_file(uploaded)
    
    # Verificar se df foi criado corretamente
    if df is not None:
        if not isinstance(df, pd.DataFrame):
            st.error("❌ Erro: O arquivo não foi carregado como um DataFrame válido")
            df = None
        elif df.empty:
            st.warning("⚠️ O arquivo carregado está vazio")
            df = None
        
except Exception as e:
    st.error(f"❌ Erro ao carregar dados: {str(e)}")
    df = None

if df is None:
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Bem-vindo ao Dynamic Stats!</h3>
        <p>Para começar, envie um arquivo CSV/Excel na barra lateral ou ative o dataset de exemplo.</p>
        <p>💡 <strong>Dica:</strong> O dataset de exemplo é perfeito para testar as funcionalidades!</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Mensagem de sucesso estilizada
if df is not None:
    st.markdown(f"""
    <div class="success-box">
        <h4>✅ Dados carregados com sucesso!</h4>
        <p><strong>{df.shape[0]}</strong> linhas × <strong>{df.shape[1]}</strong> colunas</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("❌ Erro: DataFrame não foi carregado corretamente")
    st.stop()

with st.expander("👀 Visualizar amostra (primeiras 100 linhas)"):
    if df is not None:
        st.dataframe(df.head(100), use_container_width=True)
    else:
        st.error("❌ Erro: DataFrame não disponível")

# Identificar tipos
if df is not None:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
else:
    num_cols = []
    cat_cols = []

# -------------------- Abas --------------------
aba0, aba1, aba2 = st.tabs(["📋 Resumo & Qualidade", "🔍 Exploração (PyGWalker)", "🧪 Testes Estatísticos"])

# -------------------- ABA 0: Resumo & Qualidade --------------------
with aba0:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.markdown("### 📊 Visão Geral do Dataset")
    
    # Métricas em cards
    if df is not None:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Linhas</h3>
                <h2>{df.shape[0]:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Colunas</h3>
                <h2>{df.shape[1]:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💾 Memória</h3>
                <h2>{round(df.memory_usage(deep=True).sum() / (1024**2), 3)} MB</h2>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("❌ Erro: DataFrame não disponível para exibir métricas")

    st.markdown("### 🔍 Tipos de Dados")
    if df is not None:
        st.dataframe(pd.DataFrame({"dtype": df.dtypes.astype(str)}), use_container_width=True)

        st.markdown("### ⚠️ Valores Ausentes por Coluna")
        na_table = df.isna().sum().to_frame("missing").assign(missing_pct=lambda x: 100 * x["missing"] / len(df))
        st.dataframe(na_table, use_container_width=True)

        st.markdown("### 📈 Estatísticas Descritivas")
        with st.expander("🔢 Colunas Numéricas"):
            if len(num_cols) > 0:
                st.dataframe(df.describe(include=[np.number]), use_container_width=True)
            else:
                st.info("ℹ️ Sem colunas numéricas encontradas.")
        with st.expander("🏷️ Colunas Categóricas"):
            if len(cat_cols) > 0:
                try:
                    st.dataframe(df.describe(include=["object", "category"]).fillna(""), use_container_width=True)
                except Exception:
                    st.info("ℹ️ Sem colunas categóricas encontradas.")
            else:
                st.info("ℹ️ Sem colunas categóricas encontradas.")

        st.markdown("### 🔗 Heatmap de Correlação (Colunas Numéricas)")
        if len([c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]) >= 2:
            corr = df.select_dtypes(include=[np.number]).corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, square=True, ax=ax)
            plt.title("Matriz de Correlação", fontsize=16, fontweight='bold')
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("ℹ️ É necessário pelo menos duas colunas numéricas para gerar o heatmap de correlação.")

        st.markdown("### 💾 Baixar Dados")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Baixar CSV", 
            data=csv_bytes, 
            file_name="dynamic_stats_dados.csv", 
            mime="text/csv",
            help="Clique para baixar os dados em formato CSV"
        )
    else:
        st.error("❌ Erro: DataFrame não disponível para análise")
    
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- ABA 1: PyGWalker --------------------
with aba1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.markdown("### 🚀 Construa Análises e Visualizações Interativas")
    st.markdown("""
    <div class="info-box">
        <p><strong>💡 PyGWalker</strong> é uma ferramenta poderosa que permite criar visualizações interativas e dashboards dinâmicos.</p>
        <p>🎨 Arraste e solte variáveis para criar gráficos, tabelas e análises personalizadas! Max 1000 linhas e 20 cols</p>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        # Verificar tamanho do dataset e otimizar - atenção a erros nesta sessão
        total_rows = len(df)
        total_cols = len(df.columns)
        
        st.info(f"📊 Dataset: {total_rows:,} linhas × {total_cols} colunas")
        
        # Para evitar timeout, usar amostra muito menor
        if total_rows > 1000:
            st.warning("⚠️ Dataset grande detectado. Usando amostra de 1.000 linhas para evitar timeout.")
            sample_size = min(1000, total_rows)
            df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            st.info(f"📋 Amostra: {len(df_sample):,} linhas selecionadas aleatoriamente")
        else:
            df_sample = df.copy()
        
        # Limitar número de colunas também
        if total_cols > 20:
            st.warning("⚠️ Muitas colunas detectadas. Usando apenas as primeiras 20 para melhor performance.")
            df_sample = df_sample.iloc[:, :20]
        
        # Limpar e preparar dados para PyGWalker
        st.markdown("### 🔍 Preparação dos Dados para PyGWalker")
        
        # Converter tipos de dados problemáticos
        df_clean = df_sample.copy()
        
        # Converter colunas com muitos valores únicos para string se necessário
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Verificar se é realmente categórica ou se tem muitos valores únicos
                unique_count = df_clean[col].nunique()
                if unique_count > 50:  # Se tem muitos valores únicos, pode ser problemático
                    st.warning(f"⚠️ Coluna '{col}' tem {unique_count} valores únicos. Pode causar problemas no PyGWalker.")
        
        # Verificar valores NaN
        nan_counts = df_clean.isna().sum()
        if nan_counts.sum() > 0:
            st.warning("⚠️ Valores ausentes detectados. Preenchendo com valores apropriados...")
            # Preencher NaN com valores apropriados
            for col in df_clean.columns:
                if df_clean[col].dtype in ['int64', 'float64']:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    df_clean[col] = df_clean[col].fillna('N/A')
        
        # Mostrar informações dos dados limpos
        st.info(f"📊 Dados limpos: {len(df_clean)} linhas × {len(df_clean.columns)} colunas")
        st.dataframe(df_clean.head(10), use_container_width=True)
        
        # Opção para usar PyGWalker ou visualizações alternativas
        use_pygwalker = st.checkbox("🎨 Usar PyGWalker (pode ser lento)", value=False)
        
        if use_pygwalker:
            # Reset do PyGWalker
            if st.button("🔄 Reset PyGWalker"):
                st.rerun()
            
            # Configurações do PyGWalker para melhor performance
            try:
                with st.spinner("🚀 Carregando PyGWalker... (pode demorar)"):
                    # Configuração mínima para evitar timeout
                    renderer = StreamlitRenderer(
                        df_clean,  # Usar df_clean aqui
                        debug=False,
                        use_kernel_calc=False,  # Desabilitar para melhor performance
                        show_cloud_tool=False
                    )
                    
                    # Renderizar
                    renderer.explorer()
                    
            except Exception as e:
                st.error("❌ PyGWalker não funcionou. Usando visualizações alternativas.")
                st.error(f"Erro: {str(e)}")
                st.info("💡 Dica: Tente usar um dataset menor ou verifique se o PyGWalker está atualizado")
                use_pygwalker = False
        
        if not use_pygwalker:
            # Visualizações alternativas mais rápidas
            st.markdown("### 📊 Visualizações Rápidas (Alternativa ao PyGWalker)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Análise Numérica")
                if len(num_cols) >= 2:
                    x_col = st.selectbox("Eixo X", num_cols, key="x_alt")
                    y_col = st.selectbox("Eixo Y", [c for c in num_cols if c != x_col], key="y_alt")
                    
                    if x_col and y_col:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(df_clean[x_col], df_clean[y_col], alpha=0.6)
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
                        ax.set_title(f"Dispersão: {x_col} vs {y_col}")
                        st.pyplot(fig)
                else:
                    st.info("Necessário pelo menos 2 colunas numéricas")
            
            with col2:
                st.markdown("#### 📊 Análise Categórica")
                if len(cat_cols) >= 1:
                    cat_col = st.selectbox("Coluna Categórica", cat_cols, key="cat_alt")
                    
                    if cat_col:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        value_counts = df_clean[cat_col].value_counts().head(10)
                        ax.bar(range(len(value_counts)), value_counts.values)
                        ax.set_xticks(range(len(value_counts)))
                        ax.set_xticklabels(value_counts.index, rotation=45)
                        ax.set_title(f"Distribuição: {cat_col}")
                        plt.tight_layout()
                        st.pyplot(fig)
                else:
                    st.info("Necessário pelo menos 1 coluna categórica")
            
            # Estatísticas descritivas
            st.markdown("### 📋 Estatísticas Descritivas")
            if len(num_cols) > 0:
                st.dataframe(df_clean[num_cols].describe(), use_container_width=True)
            else:
                st.info("Sem colunas numéricas para mostrar estatísticas")
    else:
        st.error("❌ Erro: DataFrame não disponível para PyGWalker")
    
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- ABA 2: Testes Estatísticos --------------------
with aba2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.markdown("### 🧪 Laboratório de Testes Estatísticos")
    st.markdown("""
    <div class="info-box">
        <p><strong>🔬 Escolha um teste estatístico</strong> e selecione as colunas relevantes dos seus dados.</p>
        <p>📊 Os resultados incluem estatísticas, valores-p e interpretações para facilitar sua análise.</p>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.error("❌ Erro: DataFrame não disponível para testes estatísticos")
        st.stop()

    teste = st.selectbox(
        "🎯 Teste Estatístico",
        (
            "Correlação (Pearson)",
            "Correlação (Spearman)",
            "t de Student (independente)",
            "t pareado",
            "Mann–Whitney U",
            "Wilcoxon pareado",
            "ANOVA one-way",
            "Kruskal–Wallis",
            "Qui-quadrado de independência",
        )
    )

    alpha = st.number_input("📊 Nível de significância (alpha)", value=0.05, min_value=0.0001, max_value=0.5, step=0.01)

    # Interface dinâmica por teste
    result = None

    if teste in ("Correlação (Pearson)", "Correlação (Spearman)"):
        if len(num_cols) < 2:
            st.warning("⚠️ É necessário pelo menos duas colunas numéricas para análise de correlação.")
        else:
            c1 = st.selectbox("📈 Variável X (numérica)", num_cols)
            c2 = st.selectbox("📉 Variável Y (numérica)", [c for c in num_cols if c != c1])
            if c1 and c2:  # Verificar se as colunas foram selecionadas
                x = df[c1].dropna()
                y = df[c2].dropna()
                n = min(len(x), len(y))
                if teste == "Correlação (Pearson)":
                    r, p = stats.pearsonr(x.iloc[:n], y.iloc[:n])
                    metodo = "Pearson"
                else:
                    r, p = stats.spearmanr(x.iloc[:n], y.iloc[:n])
                    metodo = "Spearman"
                result = {"método": metodo, "r": r, "p": p, "n": n}

    elif teste == "t de Student (independente)":
        if len(num_cols) == 0:
            st.warning("⚠️ É necessário pelo menos uma coluna numérica para este teste.")
        elif len(cat_cols) == 0:
            st.warning("⚠️ É necessário pelo menos uma coluna categórica para este teste.")
        else:
            ycol = st.selectbox("📊 Variável resposta (numérica)", num_cols)
            gcol = st.selectbox("🏷️ Grupo (categórico com 2 níveis)", cat_cols)
            if ycol and gcol:  # Verificar se as colunas foram selecionadas
                groups = df[[ycol, gcol]].dropna()
                níveis = groups[gcol].unique()
                if len(níveis) == 2:
                    g1 = groups[groups[gcol] == níveis[0]][ycol]
                    g2 = groups[groups[gcol] == níveis[1]][ycol]
                    equal_var = st.checkbox("🔍 Assumir variâncias iguais?", value=False)
                    t, p = stats.ttest_ind(g1, g2, equal_var=equal_var)
                    d = cohen_d(g1, g2)
                    # Testes de pressupostos
                    sh1 = stats.shapiro(g1) if len(g1) >= 3 else (np.nan, np.nan)
                    sh2 = stats.shapiro(g2) if len(g2) >= 3 else (np.nan, np.nan)
                    lev = stats.levene(g1, g2) if (len(g1) >= 2 and len(g2) >= 2) else (np.nan, np.nan)
                    result = {
                        "t": t,
                        "p": p,
                        "n1": len(g1),
                        "n2": len(g2),
                        "Cohen_d": d,
                        "equal_var": equal_var,
                        "grupos": list(map(str, níveis)),
                        "Shapiro_g1_W": sh1[0],
                        "Shapiro_g2_W": sh2[0],
                        "Shapiro_g1_p": sh1[1],
                        "Shapiro_g2_p": sh2[1],
                        "Levene_W": getattr(lev, "statistic", np.nan),
                        "Levene_p": getattr(lev, "pvalue", np.nan),
                    }
                else:
                    st.warning("⚠️ A coluna de grupo precisa ter exatamente 2 níveis.")

    elif teste == "t pareado":
        if len(num_cols) < 2:
            st.warning("⚠️ É necessário pelo menos duas colunas numéricas para este teste.")
        else:
            c1 = st.selectbox("📊 Medição 1 (numérica)", num_cols)
            c2 = st.selectbox("📊 Medição 2 (numérica)", [c for c in num_cols if c != c1])
            if c1 and c2:  # Verificar se as colunas foram selecionadas
                paired = df[[c1, c2]].dropna()
                t, p = stats.ttest_rel(paired[c1], paired[c2])
                d = (paired[c1].mean() - paired[c2].mean()) / paired[c1].std(ddof=1)
                result = {"t": t, "p": p, "n": len(paired), "Cohen_d_aprox": d}

    elif teste == "Mann–Whitney U":
        if len(num_cols) == 0:
            st.warning("⚠️ É necessário pelo menos uma coluna numérica para este teste.")
        elif len(cat_cols) == 0:
            st.warning("⚠️ É necessário pelo menos uma coluna categórica para este teste.")
        else:
            ycol = st.selectbox("📊 Variável resposta (numérica)", num_cols)
            gcol = st.selectbox("🏷️ Grupo (categórico com 2 níveis)", cat_cols)
            if ycol and gcol:  # Verificar se as colunas foram selecionadas
                groups = df[[ycol, gcol]].dropna()
                níveis = groups[gcol].unique()
                if len(níveis) == 2:
                    g1 = groups[groups[gcol] == níveis[0]][ycol]
                    g2 = groups[groups[gcol] == níveis[1]][ycol]
                    alt = st.selectbox("🎯 Hipótese alternativa", ["two-sided", "less", "greater"], index=0)
                    u, p = stats.mannwhitneyu(g1, g2, alternative=alt)
                    result = {"U": u, "p": p, "n1": len(g1), "n2": len(g2), "grupos": list(map(str, níveis)), "alternative": alt}
                else:
                    st.warning("⚠️ A coluna de grupo precisa ter exatamente 2 níveis.")

    elif teste == "Wilcoxon pareado":
        if len(num_cols) < 2:
            st.warning("⚠️ É necessário pelo menos duas colunas numéricas para este teste.")
        else:
            c1 = st.selectbox("📊 Medição 1 (numérica)", num_cols)
            c2 = st.selectbox("📊 Medição 2 (numérica)", [c for c in num_cols if c != c1])
            if c1 and c2:  # Verificar se as colunas foram selecionadas
                paired = df[[c1, c2]].dropna()
                alt = st.selectbox("🎯 Hipótese alternativa", ["two-sided", "less", "greater"], index=0)
                w, p = stats.wilcoxon(paired[c1], paired[c2], alternative=alt)
                result = {"W": w, "p": p, "n": len(paired), "alternative": alt}

    elif teste == "ANOVA one-way":
        if len(num_cols) == 0:
            st.warning("⚠️ É necessário pelo menos uma coluna numérica para este teste.")
        elif len(cat_cols) == 0:
            st.warning("⚠️ É necessário pelo menos uma coluna categórica para este teste.")
        else:
            ycol = st.selectbox("📊 Variável resposta (numérica)", num_cols)
            gcol = st.selectbox("🏷️ Fator (categórico com ≥2 níveis)", cat_cols)
            if ycol and gcol:  # Verificar se as colunas foram selecionadas
                groups = df[[ycol, gcol]].dropna()
                nivel_vals = [grp[ycol].values for _, grp in groups.groupby(gcol)]
                if len(nivel_vals) >= 2:
                    f, p = stats.f_oneway(*nivel_vals)
                    # eta^2 simples
                    grand_mean = groups[ycol].mean()
                    ss_between = sum([len(v)*(v.mean()-grand_mean)**2 for v in nivel_vals])
                    ss_total = ((groups[ycol]-grand_mean)**2).sum()
                    eta2 = ss_between/ss_total if ss_total > 0 else np.nan
                    result = {"F": f, "p": p, "k": len(nivel_vals), "eta2": eta2}

                    # Teste de homogeneidade (Levene)
                    try:
                        lev = stats.levene(*nivel_vals)
                        result.update({"Levene_W": lev.statistic, "Levene_p": lev.pvalue})
                    except Exception:
                        pass

                    # Pós-hoc Tukey se significativo
                    if p < alpha:
                        try:
                            tukey = pairwise_tukeyhsd(groups[ycol], groups[gcol], alpha=alpha)
                            st.markdown("##### 🔍 Pós-hoc Tukey (alpha = %.3f)" % alpha)
                            st.dataframe(pd.DataFrame(data=tukey.summary(data=False)[1:], columns=tukey.summary().data[0]), use_container_width=True)
                        except Exception as e:
                            st.info(f"ℹ️ Não foi possível executar o teste de Tukey: {e}")
                else:
                    st.warning("⚠️ O fator precisa de pelo menos 2 níveis.")

    elif teste == "Kruskal–Wallis":
        if len(num_cols) == 0:
            st.warning("⚠️ É necessário pelo menos uma coluna numérica para este teste.")
        elif len(cat_cols) == 0:
            st.warning("⚠️ É necessário pelo menos uma coluna categórica para este teste.")
        else:
            ycol = st.selectbox("📊 Variável resposta (numérica)", num_cols)
            gcol = st.selectbox("🏷️ Fator (categórico com ≥2 níveis)", cat_cols)
            if ycol and gcol:  # Verificar se as colunas foram selecionadas
                groups = df[[ycol, gcol]].dropna()
                nivel_vals = [grp[ycol].values for _, grp in groups.groupby(gcol)]
                if len(nivel_vals) >= 2:
                    h, p = stats.kruskal(*nivel_vals)
                    result = {"H": h, "p": p, "k": len(nivel_vals)}
                else:
                    st.warning("⚠️ O fator precisa de pelo menos 2 níveis.")

    elif teste == "Qui-quadrado de independência":
        if len(cat_cols) < 2:
            st.warning("⚠️ É necessário pelo menos duas colunas categóricas para este teste.")
        else:
            c1 = st.selectbox("🏷️ Variável 1 (categórica)", cat_cols)
            c2 = st.selectbox("🏷️ Variável 2 (categórica)", [c for c in cat_cols if c != c1])
            if c1 and c2:  # Verificar se as colunas foram selecionadas
                table = pd.crosstab(df[c1], df[c2])
                chi2, p, dof, _ = stats.chi2_contingency(table)
                v = cramers_v(chi2, table.values.sum(), table.shape[0], table.shape[1])
                result = {"chi2": chi2, "p": p, "gl": dof, "Cramer_V": v}
                with st.expander("📊 Tabela de contingência"):
                    st.dataframe(table, use_container_width=True)

    # Exibir resultado
    if result is not None:
        st.markdown("### 📊 Resultados do Teste")
        
        # Card de resultado estilizado
        col1, col2 = st.columns([2, 1])
        with col1:
            st.json(result)
        with col2:
            sig = result.get("p", np.nan) < alpha if "p" in result else None
            if sig is not None:
                if sig:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                                padding: 1rem; border-radius: 10px; text-align: center;">
                        <h4>🎯 Resultado Significativo!</h4>
                        <p><strong>Hipótese nula REJEITADA</strong></p>
                        <p>p < alpha (α = {:.3f})</p>
                    </div>
                    """.format(alpha), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                                padding: 1rem; border-radius: 10px; text-align: center;">
                        <h4>📊 Resultado Não Significativo</h4>
                        <p><strong>Hipótese nula NÃO rejeitada</strong></p>
                        <p>p ≥ alpha (α = {:.3f})</p>
                    </div>
                    """.format(alpha), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <p>💡 <strong>Dica:</strong> Use a aba 'Exploração (PyGWalker)' para montar gráficos interativos e, se desejar, retorne aqui para executar testes estatísticos.</p>
        <p>🚀 <strong>Dynamic Stats</strong> - Sua ferramenta completa para análise de dados!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

