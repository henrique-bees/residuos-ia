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

## 4. Estado atual (atualizado em 2026-11-17)

**Ambiente**
- Repositório git inicializado localmente (ainda sem remoto no GitHub). Venv em `.venv/`
  (Python 3.14) com `requirements.txt` instalado; GPU CUDA disponível neste ambiente.
- Hook de projeto exige aprovação explícita (`.claude/commit-approved`, consumido a cada uso)
  antes de qualquer `git commit` — ver skill `commit-por-etapas`.
- `pytest testes -q` funciona direto agora (adicionado `pytest.ini` com `pythonpath = .`;
  antes só funcionava com `python -m pytest`).

**Pronto e testado**
- Todos os módulos implementados e integrados; o app abre e responde (testado com `streamlit.testing`).
- **Tarefa 1 (visão computacional) — CONCLUÍDA com dados reais do TrashNet:**
  CNN MobileNetV2 (teste): acurácia **0,871**, F1-macro **0,865**. Baseline HOG+SVM (teste):
  acurácia **0,700**, F1-macro **0,697**. Splits em `dados/splits/*.csv` (commitados). Modelos
  e imagens do dataset ficam fora do git (`modelos/`, `dados/trashnet/`).
- **Tarefa 2 (PLN) — PARCIALMENTE feita:** `dados/intencoes.csv` ampliado de ~145 para
  **296 frases** (146 `origem=equipe` + 150 `origem=gerado_ia`, geradas por mim, não por
  colegas reais — está marcado na coluna `origem` do CSV para ser honesto sobre a origem).
  F1-macro no teste subiu de 0,696 para **0,935** (melhor modelo agora é SVM linear calibrado,
  CV 0,877 ± 0,061). **Ressalva importante:** frases geradas por IA tendem a ser mais
  homogêneas que frases reais, então parte do ganho pode estar inflado — o ideal é substituir
  as linhas `gerado_ia` por frases reais de colegas (gírias, erros de digitação variados) e
  re-rodar `python -m src.pln.intencoes` para ver se o F1 real se mantém.
- **Bug real encontrado e corrigido (2026-11-18), achado pelo Henrique testando o app:**
  frases com "como" + "reciclar" (ex. "como reciclar isso?") caíam na intenção `reciclavel`
  (sim/não) em vez de `como_descartar` (instruções), porque os exemplos de treino de
  `como_descartar` só usavam "descartar/separar/jogar fora", nunca "reciclar". Corrigido
  adicionando 10 frases `como_descartar` com "como...reciclar" e re-treinando.
- **Bug real encontrado e corrigido (2026-11-18):** o app não tinha memória de conversa — se
  o usuário dizia o material numa mensagem ("isso é plástico") e perguntava algo na mensagem
  seguinte ("como reciclo isso?"), o material citado antes era esquecido (cada pergunta era
  processada isoladamente). Corrigido com `com_memoria_de_material()` em
  `src/pln/entidades.py` + estado de sessão no `app.py`: se a pergunta atual não cita
  material, reaproveita o último material citado por texto na mesma conversa (reiniciado ao
  trocar a foto). Testado com `testes/test_pln.py`.
- **Tarefa 4 (exemplos) — CONCLUÍDA:** 8 exemplos reais em `resultados/exemplos/` (imagem do
  split de teste + pergunta → resposta do agente + regras), gerados pelo pipeline completo
  (CNN real + PLN real + agente + A*). Inclui um erro real da CNN (papelão confundido com
  papel) — útil para a seção de limitações do artigo.
- **Tarefa 5 (artigo) — RASCUNHO pronto, não compilado:** `artigo/main.tex` +
  `artigo/referencias.bib` com os números reais acima. Não há LaTeX instalado neste ambiente
  (precisaria de `sudo apt install texlive-...`, que exige senha interativa), então só foi
  validado manualmente (braces/ambientes balanceados, figuras existem nos caminhos citados).
  **Falta:** compilar no Overleaf e conferir se cabe em 6 páginas, preencher integrantes e
  link do GitHub (ver pendências em `artigo/README.md`).

**Ainda NÃO feito (precisa de ação humana, não dá pra automatizar com integridade)**
- **Tarefa 2 completa:** trocar as 140 frases `gerado_ia` por frases reais de colegas.
- **Tarefa 3 (teste de campo):** precisa de ~20 fotos reais por classe tiradas pela equipe
  em `dados/teste_campo/<classe>/`, e do script `avaliar_campo.py` (ainda não escrito).
- **Tarefa 6:** push pro GitHub já feito (`https://github.com/henrique-bees/residuos-ia`,
  privado) e nome do integrante já preenchido (Henrique de Moraes Belo Lopes, equipe de 1
  pessoa). Falta só o link do Overleaf e (opcional) substituir os pontos de coleta simulados
  por dados oficiais de Maceió.

## 5. Próximas tarefas (em ordem)

### Tarefa 1: Treinar e avaliar a Visão Computacional — ✅ CONCLUÍDA (2026-11-17)
CNN 0,871/0,865 (acurácia/F1), baseline 0,700/0,697. Ver seção 4.

### Tarefa 2: Melhorar o dataset de PLN — ⚠️ PARCIAL (2026-11-17)
`dados/intencoes.csv` já tem ≥ 40 frases/intenção, mas 140 delas são `origem=gerado_ia`
(geradas por mim), não frases reais de colegas. **O que falta de verdade:**
- Substituir as linhas `origem=gerado_ia` por frases reais coletadas com colegas (gírias,
  erros de digitação, jeitos diferentes de perguntar a mesma coisa).
- Rodar `python -m src.pln.intencoes` de novo e comparar o F1 com o 0,943 atual — se cair
  bastante, é sinal de que o modelo anterior estava mais otimista do que deveria por causa
  da homogeneidade das frases geradas; registrar isso no artigo (Seção IV já tem a ressalva).
- Opcional: comparar com embeddings (`sentence-transformers`, ex. `paraphrase-multilingual-MiniLM-L12-v2`)
  + regressão logística, como experimento extra.

### Tarefa 3: Teste em campo (fortalece Experimentos e Limitações) — ⚠️ script pronto, falta as fotos
- O script `src/visao_computacional/avaliar_campo.py` já está pronto e testado (com imagens
  do TrashNet, depois removidas — não ficou nenhum resultado fictício no repo). Ele lê
  `dados/teste_campo/<classe>/*.jpg`, avalia o modelo salvo, gera
  `resultados/metricas_cnn_campo.json` + matriz de confusão, e já calcula a taxa de recusa
  (confiança < 60%) e a acurácia entre as respostas aceitas.
- **Falta só:** a equipe fotografar ~20 resíduos reais por classe (fundo comum, celular),
  colocar em `dados/teste_campo/<classe>/` e rodar `python -m src.visao_computacional.avaliar_campo`.
- Depois, comparar com o resultado do TrashNet (queda esperada por mudança de domínio) e
  atualizar a Seção IV do artigo com esse número.

### Tarefa 4: Exemplos de entrada e saída — ✅ CONCLUÍDA (2026-11-17)
8 exemplos reais em `resultados/exemplos/` (script usado: ver histórico do git; pode ser
regerado/ampliado facilmente adaptando a lista `CASOS` do script que gerou os arquivos atuais).

### Tarefa 5: Artigo (IEEE, ≤ 6 páginas) — ⚠️ RASCUNHO pronto, falta compilar
`artigo/main.tex` + `artigo/referencias.bib` escritos com os números reais de `resultados/`.
**Falta:** compilar no Overleaf (ambiente de desenvolvimento não tem LaTeX) e conferir se
cabe em 6 páginas; preencher integrantes e link do GitHub (marcados como `preencher` no
`.tex`). Ver pendências detalhadas em `artigo/README.md`.

### Tarefa 6: Acabamento — ⚠️ PARCIAL (2026-11-17)
- ✅ Integrante preenchido (Henrique de Moraes Belo Lopes) no README e no `artigo/main.tex`.
- ✅ Repositório criado no GitHub e `git push` feito: `https://github.com/henrique-bees/residuos-ia`
  (público).
- ✅ Link do Overleaf (modo Viewer) adicionado no README:
  https://www.overleaf.com/project/6afce9ccfddd89d11d81d8ae/share#786d039eed9eab936c4c1d4e0a29961643fc71ad89f190aa
- **Falta:** confirmar no Overleaf que o artigo compila sem erro e cabe em 6 páginas.
- **Falta:** conferir que um clone limpo + os comandos do README reproduz tudo (não testado
  ainda — só foi validado rodando os comandos diretamente neste ambiente).
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
- Dataset de intenções: 146 frases da equipe + 140 geradas por IA (`origem` no CSV); o F1 de
  0,94 pode estar inflado pela homogeneidade das frases geradas — ideal é trocá-las por frases
  reais de colegas antes de reportar esse número como definitivo no artigo.
- Mapa e pontos de coleta simulados; distância = linha reta × fator viário de 1,3.
- Resíduos especiais (pilhas, eletrônicos, óleo, lâmpadas) fora das classes.
