# Placebo sazonal 2023/24 — pausa FIFA de novembro

Este relatório usa partidas **reais** de 2023/24. O marcador `treated_world_cup`
indica quem integrou a lista oficial de Qatar 2022; como a Copa já havia
terminado, os coeficientes são uma falsificação de trajetória, não efeitos da Copa.

- Corte pré-pausa: 12/11/2023.
- Retorno da janela internacional: 24/11/2023.
- Janelas: oito jogos antes, quatro agudos depois e quatro consolidados depois.
- Reconstrução dos dados brutos: `python scripts/data/download_transfermarkt_placebo_2023_24.py`.
- Reconstrução da análise: `python scripts/analysis/run_placebo_2023_24.py`.
- População ponderada: 953 jogadores (245 futuros convocados e 708 controles).
- Ridge selecionado: 2; ESS tratado/controle: 159.02/267.11.
- Após calibração, máximo |SMD|: 1.43e-13; máximo KS: 0.1006; razão de variância: 0.820–1.319.
- Gates de ponderação: não aprovados; resultados são somente diagnósticos. O ESS dos controles foi 37.7% (gate: 50%) e o KS máximo foi 0.1006 (gate: 0,10).
- Efeito placebo agudo em `minutes_share`: 0.0046 (IC95% -0.0562 a 0.0654; p=0.8820).
- Efeito placebo consolidado: -0.0680 (IC95% -0.1333 a -0.0027; p=0.0411).

## Limites

Esta falsificação cobre `minutes_share`, não npxG+xA, lesões ou dados
físicos: os snapshots locais de Understat e lesões para 2023/24 não estão
incluídos no pipeline. Os pesos foram recalculados com covariáveis pré-pausa disponíveis
(idade, valor, utilização, titularidade, força de clube/adversário e trajetória
de minutos), sem reutilizar pesos de 2022/23. A elegibilidade nacional prioriza
a tabela auditada de 2022 quando o atleta está presente nela; para demais atletas,
usa cidadania primária do Transfermarkt e deve ser lida como proxy.

Não há holdout pré-pausa homogêneo nesta temporada: em 12/11/2023, a
Bundesliga tinha somente 11 rodadas concluídas. Assim, todos os oito jogos
pré-pausa disponíveis para a especificação foram usados na calibração; o
placebo não acrescenta um teste independente de pré-tendência. Além disso, a
coorte observável em 2023/24 não é idêntica à de 2022/23 (314 convocados), e o
snapshot público baixado pode diferir do snapshot histórico usado na análise
principal.
