# Fase 7 — robustez, falsificações e exploração limitada

Implementação concluída em 12/07/2026. Todos os resultados abaixo usam o painel
e os pesos congelados das fases anteriores; métodos que mudam o estimando são
identificados explicitamente como sensibilidades.

## Resultado executivo

O padrão nos dados é compatível com ausência de efeito agudo claro e com um
aumento tardio de utilização, mas o aumento consolidado não é robusto a pequenas
violações de tendências paralelas. A formulação permitida continua sendo
**associação longitudinal ajustada**, não efeito causal estabelecido.

## Janelas e amostras

| Janela simétrica | Efeito em `minutes_share` | IC 95% | p |
|---|---:|---:|---:|
| 4 jogos | 0,0209 | [-0,0311; 0,0729] | 0,430 |
| 6 jogos | 0,0413 | [-0,0076; 0,0903] | 0,098 |
| 8 jogos | 0,0534 | [0,0085; 0,0983] | 0,020 |

O resultado cresce com a janela, sugerindo padrão tardio, mas não autoriza
selecionar oito jogos por significância. No subconjunto com pelo menos 360
minutos pré-Copa, o efeito agudo é 0,0603 (p = 0,0345, não ajustado para a matriz
de especificações). Essa diferença pode refletir seleção de jogadores mais
estabelecidos.

## Métodos de peso e populações

- overlap principal: efeito agudo 0,0233;
- matching 1:3: 0,0302, porém com desequilíbrio residual conhecido;
- sem pesos: 0,0389;
- same-nation support: 0,0339;
- entropy ATT: 0,0881, mas o método tem ESS de controles de apenas 21% e pesos
  extremos; não deve ser usado como resultado principal;
- truncamentos de 95% e 99% reproduzem o resultado principal.

O leave-one-out mostra efeito agudo entre -0,0020 e 0,0442 ao retirar uma liga,
e efeito consolidado entre 0,0682 e 0,1029. Nenhuma liga, clube ou seleção muda
isoladamente o sinal consolidado, mas isso não resolve o problema de tendência.

## Tendências paralelas e confundimento

A análise transparente inspirada em HonestDiD, mas não idêntica ao pacote
oficial, usa as violações prévias observadas como escala. Na restrição de
suavidade, `M = 0,25` já faz o limite inferior do efeito consolidado cruzar zero.
Na restrição de magnitude relativa, isso ocorre antes ou em `M = 1,0`.

O valor de robustez aproximado é 0,0287 para o efeito agudo e 0,0917 para o
consolidado. São diagnósticos aproximados para HDFE com cluster, não resultados
do `sensemakr` oficial.

## Dose e heterogeneidade

Entre os 314 convocados, as categorias de minutos na Copa têm tamanhos 20, 56,
110 e 128. Nenhum coeficiente de dose permanece significativo após Benjamini-
Hochberg. Não há evidência robusta de gradiente monotônico.

As diferenças defesa versus ataque e histórico prévio de lesão sobrevivem à BH
na família exploratória. Como são interações exploratórias e o efeito médio é
sensível, devem aparecer apenas como hipóteses para estudos futuros, sem narrativa
confirmatória de subgrupos.

## Limitações de cobertura

- jogos pós-Copa 9 e 10 não existem no painel;
- o limiar de 90 minutos exige reconstruir a população e os pesos;
- o jogo pré -15 é incompleto e o -16 não existe;
- não há painel local das temporadas 2021/22 ou 2023/24;
- near-miss e Big Five amplo exigem novo desenho de suporte e reponderação.

Essas especificações estão registradas em `coverage_matrix.csv` e não foram
preenchidas com extrapolação.

## Arquivos

- `window_sensitivity.csv`: janelas 4/6/8;
- `sample_weight_sensitivity.csv`: pesos, truncamentos e populações;
- `pretrend_length_sensitivity.csv`: histórias prévias alternativas;
- `leave_one_out.csv`: ligas, clubes e seleções;
- `parallel_trends_sensitivity.csv`: limites de violações de tendência;
- `unobserved_confounding_sensitivity.csv`: valores de robustez aproximados;
- `dose_response_associational.csv`: dose entre convocados;
- `heterogeneity_interactions.csv`: diferenças formais com BH;
- `coverage_matrix.csv`, `phase7_gate_summary.csv` e `phase7_qa.csv`: auditoria;
- `robustness_manifest.csv`: hashes e linhagem.

## Decisão para a Fase 8

Redigir o paper como estudo observacional de associação ajustada. Apresentar o
efeito agudo nulo/impreciso como resultado principal, o padrão consolidado como
sensível à tendência e dose/heterogeneidade como análises exploratórias.
