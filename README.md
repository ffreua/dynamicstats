# 🚀 Dynamic Stats

**Análise Estatística Dinâmica e Interativa**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

---

## 📋 Descrição

**Dynamic Stats** é uma ferramenta web completa e interativa para análise estatística, desenvolvida com Python e Streamlit. Oferece uma interface moderna e intuitiva para exploração de dados, testes estatísticos robustos e visualizações interativas através do PyGWalker.

## ✨ Características Principais

- 📊 **Análise Exploratória Avançada** - Resumos estatísticos, qualidade de dados e correlações
- 🔬 **Testes Estatísticos Robustos** - t-tests, ANOVA, correlações, testes não-paramétricos
- 📈 **Visualizações Interativas** - PyGWalker para dashboards dinâmicos
- 🎨 **Interface Moderna** - Design responsivo com gradientes e componentes estilizados
- 📁 **Suporte Multi-formato** - CSV, Excel (.xls, .xlsx, .xlsm)
- 💾 **Exportação de Dados** - Download dos dados processados
- 📱 **Responsivo** - Funciona em desktop e dispositivos móveis

## 🛠️ Tecnologias Utilizadas

- **Backend**: Python 3.8+
- **Framework Web**: Streamlit
- **Análise de Dados**: Pandas, NumPy
- **Estatística**: SciPy, StatsModels
- **Visualização**: Matplotlib, Seaborn, PyGWalker
- **Interface**: CSS personalizado com gradientes

## 📦 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip ou conda

### Passos de Instalação

1. **Clone o repositório**
   ```bash
   git clone https://github.com/seu-usuario/dynamic-stats.git
   cd dynamic-stats
   ```

2. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

   Ou usando conda:
   ```bash
   conda install --file requirements.txt
   ```

3. **Execute a aplicação**
   ```bash
   streamlit run app.py
   ```

4. **Acesse no navegador**
   ```
   http://localhost:8501
   ```

## 🚀 Como Usar

### 1. Carregamento de Dados
- **Dataset de Exemplo**: Ative o toggle para usar dados de demonstração
- **Upload de Arquivo**: Envie arquivos CSV ou Excel (até 200MB)
- **Formatos Suportados**: .csv, .xls, .xlsx, .xlsm

### 2. Análise Exploratória (Aba 1)
- **Visão Geral**: Estatísticas descritivas e qualidade dos dados
- **Correlações**: Heatmap de correlação para variáveis numéricas
- **Valores Ausentes**: Análise de dados faltantes por coluna
- **Exportação**: Download dos dados processados

### 3. Exploração Interativa (Aba 2)
- **PyGWalker**: Interface drag-and-drop para visualizações
- **Gráficos Dinâmicos**: Crie dashboards interativos
- **Análises Personalizadas**: Explore seus dados de forma intuitiva

### 4. Testes Estatísticos (Aba 3)
- **Correlações**: Pearson e Spearman
- **Testes de Diferenças**: t-tests, Mann-Whitney, Wilcoxon
- **ANOVA**: One-way ANOVA com pós-hoc Tukey
- **Testes Não-paramétricos**: Kruskal-Wallis, Qui-quadrado

## 📊 Testes Estatísticos Disponíveis

| Teste | Descrição | Variáveis |
|-------|-----------|-----------|
| **Correlação Pearson** | Correlação linear entre duas variáveis numéricas | 2 numéricas |
| **Correlação Spearman** | Correlação de postos entre duas variáveis | 2 numéricas |
| **t-test Independente** | Comparação de médias entre dois grupos | 1 numérica + 1 categórica (2 níveis) |
| **t-test Pareado** | Comparação de médias em medidas repetidas | 2 numéricas |
| **Mann-Whitney U** | Teste não-paramétrico para dois grupos | 1 numérica + 1 categórica (2 níveis) |
| **Wilcoxon Pareado** | Teste não-paramétrico para medidas repetidas | 2 numéricas |
| **ANOVA One-way** | Comparação de médias entre múltiplos grupos | 1 numérica + 1 categórica (≥2 níveis) |
| **Kruskal-Wallis** | ANOVA não-paramétrica | 1 numérica + 1 categórica (≥2 níveis) |
| **Qui-quadrado** | Teste de independência entre variáveis categóricas | 2 categóricas |

## 📁 Estrutura do Projeto

```
Dynamic_Stats/
├── app.py              # Aplicação principal
├── requirements.txt    # Dependências Python
├── README.md          # Este arquivo
└── .gitignore         # Arquivos a serem ignorados pelo Git
```

## 🔧 Configuração

### Variáveis de Ambiente (Opcional)
```bash
# Para personalizar a aplicação
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Personalização do Tema
O CSS personalizado pode ser modificado no arquivo `app.py` na seção de estilos.

## 📈 Recursos Avançados

- **Cache de Dados**: Otimização de performance com `@st.cache_data`
- **Validação de Dados**: Detecção automática de tipos e formatos
- **Testes de Pressupostos**: Verificação de normalidade e homogeneidade
- **Tamanho do Efeito**: Cálculo de Cohen's d e outras métricas
- **Pós-hoc Automático**: Teste de Tukey para ANOVA significativa

## 🤝 Contribuindo

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto é de distribuição gratuita para uso pessoal
ATENÇÃO: Este aplicativo não é aprovado para analise estatística. Confira sempre os resultados
Se usar o WebApp você aceita os termos acima.

## 👨‍💻 Desenvolvedor

**Dr Fernando Freua**

- 🎓 Neurologista entusiasta em programação
- 🐍 Acha que entende um pouco de Python e análise de dados, mas entende pouco!
- 📊 Quer saber estatística! mas tá difícil!
- 🚀 Criador do Dynamic Stats (com ajuda do GPT 5!)

## 🙏 Agradecimentos

- **Streamlit** - Framework web incrível - uso muito!
- **PyGWalker** - Visualizações interativas poderosas
- **Comunidade Python** - Bibliotecas e ferramentas de qualidade
- **Contribuidores** - Todos que ajudaram ou ajudarão a melhorar o projeto

## 📞 Suporte

- **Email**: fernando.freua@hc.fm.usp.br

## 🔄 Histórico de Versões

- **v2.0** - Interface redesenhada, novos testes estatísticos
- **v1.0** - Versão inicial com funcionalidades básicas

---

<div align="center">

**⭐ Se este projeto te ajudou, considere dar uma estrela! ⭐**

**Made with ❤️ by Dr Fernando Freua**

</div>
