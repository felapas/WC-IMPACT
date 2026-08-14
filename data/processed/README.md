# Dados processados — Fase 3

Arquivos gerados por `scripts/analysis/build_analytic_panel.py`. Não editar os
CSVs manualmente.

O painel principal usa uma linha por jogador e partida do clube, incluindo zeros
para reservas não utilizados e jogadores ativos fora da súmula. A elegibilidade
usa somente baseline pré-Copa e vínculo com o clube; minutos pós-Copa não são
usados para selecionar a amostra.

Veja `data_dictionary.csv` para a unidade de observação, tipo e descrição de
cada coluna, e `panel_qa_summary.csv` para os controles de qualidade. Exceções
conhecidas das fontes permanecem sem imputação e recebem flags explícitas no
painel jogador-partida.
