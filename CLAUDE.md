# CLAUDE.md — Contexto e instruções de continuação

> Este arquivo dá ao Claude Code o contexto do projeto. Leia inteiro antes de mexer no código.
> Idioma do projeto: **português (Brasil)**: código, comentários, mensagens, README e artigo.

## 1. O que é este projeto

Projeto Integrador Final da disciplina **Inteligência Artificial** (AFYA • UNIMA Maceió,
Prof. Icaro Ferreira). Equipe de 3 a 5 pessoas, vale 100 pontos.

**Tema escolhido: B — Identificação e classificação de resíduos.**
O usuário envia a **foto de um resíduo** e faz uma **pergunta em linguagem natural**. O sistema
identifica o material, diz se é reciclável, como preparar o descarte, a cor da lixeira e o ponto de
coleta mais próximo em Maceió.

### Entregas exigidas
1. **Repositório GitHub** com código, dependências, instruções, exemplos, testes e resultados.
2. **Artigo no Overleaf**: formato IEEE Conference, 2 colunas, **até 6 páginas incluindo referências**.
   Seções: resumo (150–200 palavras), Introdução, Fundamentação/Trabalhos relacionados,
   Metodologia, Resultados e discussão, Conclusão/Trabalhos futuros, Referências.
3. Demonstração funcional (ou vídeo + código executável).

### Requisitos obrigatórios (e onde cada um é atendido)
| Requisito | Pontos | Implementação |
|---|---|---|
| Definição do problema e arquitetura | 10 | README + diagrama Mermaid |
| Machine Learning | 15 | CNN com transfer learning + baseline HOG/SVM + classificador de intenções |
| Visão Computacional | 15 | Classificação de imagens em 6 classes (TrashNet) |
| PLN | 15 | Reconhecimento de intenções + extração de entidades (material, bairro) |
| Agentes / busca / decisão | 10 | Agente com sistema de regras (R0–R16) + A* no grafo de bairros |
| Integração e funcionamento | 10 | `app.py`: imagem e pergunta no mesmo fluxo |
| Experimentos e avaliação | 10 | Métricas, matrizes de confusão, curvas, comparação de buscas |
| GitHub e reprodutibilidade | 5 | Semente 42, splits estratificados, comandos no README |
| Artigo | 10 | A fazer |

**O professor penaliza:** três códigos isolados sem integração; um campo de texto sem processamento
contar como PLN; citar um algoritmo sem que ele tenha função real; usar só APIs prontas sem
avaliação; **resultados inventados**.

## 2. Regras importantes para o Claude Code

- **Nunca invente números.** Toda métrica no README ou no artigo tem que vir de um arquivo em
  `resultados/` gerado por um script deste repositório. Se o número ainda não existe, deixe
  `_preencher_` e diga isso.
- **Não use dados do PACE/CombatIQ nem de nenhum projeto de trabalho do Henrique.** São
  confidenciais e não têm relação com este projeto.
- O mapa (`dados/mapa_maceio.json`) é **simulado**: coordenadas aproximadas e pontos de coleta
  fictícios (marcados com "(simulado)"). Isso tem que continuar explícito no README e no artigo,
  a menos que seja trocado por dados oficiais com fonte citada.
- Não publique tokens, senhas ou dados pessoais. Dataset e modelos ficam fora do git (ver `.gitignore`).
- Mantenha a estrutura de pastas sugerida pelo enunciado (`src/visao_computacional`, `src/pln`,
  `src/machine_learning`, `src/decisao`, `dados`, `notebooks`, `resultados/graficos`,
  `resultados/exemplos`, `testes`, `artigo`).
- Depois de qualquer mudança, rode `pytest testes -q`. Hoje são 89 testes passando.
- Se o repositório estiver em git, faça commits pequenos, por etapa.

## 3. Arquitetura atual

```
Foto ──► Visão (MobileNetV2) ──► classe + confiança + probabilidades ─┐
                                                                      ├─► Agente (regras) ──► Resposta
Pergunta ─► PLN: intenção (TF-IDF + NB/LR/SVM) + entidades (léxico) ──┘          │            + rastro de regras
                                                                                 └─► A* (onde descartar)
```

### Arquivos principais
| Arquivo | Função |
|---|---|
| `src/config.py` | Caminhos, classes (`cardboard, glass, metal, paper, plastic, trash`), nomes em PT, limiares |
| `src/visao_computacional/dataset.py` | Baixa o TrashNet, descarta imagens corrompidas, divide 70/15/15 estratificado → `dados/splits/*.csv` |
| `src/visao_computacional/modelo.py` | MobileNetV2 do ImageNet, data augmentation, liberação de camadas para fine-tuning |
| `src/visao_computacional/treinar.py` | Treino em 2 fases (extração de características → fine-tuning), salva o melhor pela validação, avalia no teste |
| `src/visao_computacional/classificador.py` | Inferência: `ClassificadorResiduos.classificar(img) → ResultadoVisao` (classe, confiança, probabilidades, margem) |
| `src/machine_learning/baseline_svm.py` | HOG + histograma HSV + SVM com GridSearch (comparação ML clássico x CNN) |
| `src/machine_learning/metricas.py` | Relatório (acurácia, F1-macro, por classe), matriz de confusão, curvas de treino |
| `src/pln/preprocessamento.py` | Minúsculas, remoção de acentos e pontuação, expansão de abreviações ("vc", "pq") |
| `src/pln/intencoes.py` | TF-IDF (palavras 1–2 + caracteres 2–5) e comparação LR × Naive Bayes × SVM linear com CV de 5 folds |
| `src/pln/entidades.py` | Léxico de materiais (termo mais longo vence: "papel engordurado" → rejeito) e detecção de bairro |
| `src/decisao/regras.py` | Base de conhecimento: reciclável?, cor da lixeira (CONAMA 275/2001), orientações de preparo |
| `src/decisao/grafo.py` | Grafo de bairros; A* com heurística admissível (haversine até o objetivo mais próximo), custo uniforme e BFS |
| `src/decisao/agente.py` | `AgenteDescarte.decidir(...)`: regras R0–R16. O texto tem prioridade sobre a visão; confiança < 0,60 → pede nova foto; margem < 0,15 → ressalva |
| `src/decisao/experimento_busca.py` | Compara A*, custo uniforme e BFS em todas as origens e materiais |
| `app.py` | Streamlit: upload ou câmera, gráfico de probabilidades, chat e a seção "Como o agente decidiu" |
| `dados/intencoes.csv` | ~145 frases em 7 intenções (escritas pela equipe) |
| `testes/` | `test_pln.py` e `test_decisao.py` (otimalidade do A*, admissibilidade da heurística, regras do agente) |

Intenções: `identificar`, `reciclavel`, `como_descartar`, `cor_lixeira`, `onde_descartar`, `saudacao`, `fora_escopo`.

## 4. Estado atual

**Pronto e testado**
- Todos os módulos implementados e integrados; o app abre e responde (testado com `streamlit.testing`).
- O pipeline de treino da CNN e do baseline roda do começo ao fim (verificado com imagens sintéticas,
  que já foram apagadas; nenhum resultado delas foi mantido).
- Resultados reais já gerados:
  - Intenções: Naive Bayes é o melhor, com F1-macro na CV de **0,811 ± 0,041** (LR 0,803; SVM 0,785).
    No teste (25%): acurácia **0,757**, F1-macro **0,696**. Arquivo: `resultados/metricas_intencoes.json`.
  - Busca: A* examina em média **3,13** nós e acha o caminho ótimo em 100% dos casos. Custo uniforme:
    4,76 nós, 100%. BFS: 4,60 nós, 74,5% ótimo. Arquivos: `resultados/busca_resumo.csv` e `busca_comparacao.csv`.

**Ainda NÃO feito**
- A CNN e o baseline ainda não foram treinados no TrashNet real (o download estava bloqueado no
  ambiente onde o código foi escrito).

## 5. Próximas tarefas (em ordem)

### Tarefa 1: Treinar e avaliar a Visão Computacional
```bash
pip install -r requirements.txt
python -m src.visao_computacional.dataset --baixar --dividir
python -m src.visao_computacional.treinar --epocas 8 --epocas-ft 6
python -m src.machine_learning.baseline_svm
```
- Se o download falhar, baixe `data/dataset-resized.zip` de https://github.com/garythung/trashnet,
  coloque em `dados/` e rode `--baixar` de novo (o script reaproveita o zip local).
- Sem GPU, considere criar `notebooks/treino_colab.ipynb`, que clona o repositório e roda os mesmos comandos no Colab.
- Faça commit de `dados/splits/*.csv` (reprodutibilidade). Não faça commit de modelos nem de imagens.
- Preencha a tabela de resultados do README com os valores de `resultados/metricas_cnn.json` e `metricas_baseline_svm.json`.

### Tarefa 2: Melhorar o dataset de PLN
- O F1 de 0,70 no teste é baixo por causa do tamanho do dataset. Amplie `dados/intencoes.csv` para
  pelo menos 40 frases por intenção, de preferência frases reais coletadas com colegas (variar a
  escrita, incluir gírias e erros de digitação).
- Rode `python -m src.pln.intencoes` de novo e atualize o README.
- Opcional: comparar com embeddings (`sentence-transformers`, ex. `paraphrase-multilingual-MiniLM-L12-v2`)
  + regressão logística, como experimento extra.

### Tarefa 3: Teste em campo (fortalece Experimentos e Limitações)
- A equipe fotografa cerca de 20 resíduos reais por classe (fundo comum, celular).
- Crie `dados/teste_campo/<classe>/` e um script `src/visao_computacional/avaliar_campo.py` que
  avalia o modelo salvo nessas fotos e gera `resultados/metricas_cnn_campo.json` + matriz de confusão.
- Compare com o resultado do TrashNet (queda esperada por mudança de domínio).
- Avalie também a regra de confiança: em quantos erros da CNN o agente pediu nova foto
  (taxa de recusa × acurácia entre as respostas aceitas). Isso dá um bom gráfico para o artigo.

### Tarefa 4: Exemplos de entrada e saída
- Gere 6–10 exemplos (imagem + pergunta → resposta + regras) e salve em `resultados/exemplos/`
  (PNG ou Markdown). O enunciado exige exemplos de entrada e saída.

### Tarefa 5: Artigo (IEEE, ≤ 6 páginas)
- Crie `artigo/main.tex` com `\documentclass[conference]{IEEEtran}` e `artigo/referencias.bib`.
- Use só números de `resultados/`. Figuras: diagrama da arquitetura, curvas da CNN, matrizes de
  confusão (CNN e intenções), gráfico de nós expandidos. Tabelas: CNN × baseline, comparação dos
  classificadores de intenção, A* × custo uniforme × BFS.
- Inclua o link do GitHub. Linguagem técnica e impessoal.
- Referências já listadas no README: Thung & Yang (2016), Sandler et al. (2018, MobileNetV2),
  Hart, Nilsson & Raphael (1968, A*), Dalal & Triggs (2005, HOG), CONAMA 275/2001.
  Verifique cada uma antes de citar.

### Tarefa 6: Acabamento
- Preencher integrantes e o link do Overleaf no README.
- Conferir que um clone limpo + os comandos do README reproduz tudo.
- Opcional: substituir os pontos simulados por ecopontos oficiais de Maceió, citando a fonte.

## 6. Comandos úteis
```bash
pytest testes -q                             # testes
python -m src.pln.intencoes                  # re-treina intenções
python -m src.decisao.experimento_busca      # re-roda experimento de busca
streamlit run app.py                         # interface
```

## 7. Limitações conhecidas (devem aparecer no artigo)
- TrashNet: fundo branco, um objeto por foto e cerca de 2.500 imagens → mudança de domínio em fotos reais.
- Dataset de intenções pequeno e escrito pela equipe.
- Mapa e pontos de coleta simulados; distância = linha reta × fator viário de 1,3.
- Resíduos especiais (pilhas, eletrônicos, óleo, lâmpadas) fora das classes.
