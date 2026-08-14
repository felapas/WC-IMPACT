# Placebo sazonal 2021/22 — pausa FIFA de novembro

Este relatório usa partidas **reais** de 2021/22. O marcador `treated_world_cup`
indica quem integraria a lista oficial de Qatar 2022; como a Copa ainda não
ocorrera, os coeficientes são uma falsificação de trajetória, não efeitos da Copa.

- Corte pré-pausa: 07/11/2021.
- Retorno da janela internacional: 19/11/2021.
- Janelas: oito jogos antes, quatro agudos depois e quatro consolidados depois.
- Reconstrução dos dados brutos: `python scripts/data/download_transfermarkt_placebo_2021_22.py`.
- Reconstrução da análise: `python scripts/analysis/run_placebo_2021_22.py`.
- População ponderada: 965 jogadores (267 futuros convocados e 698 controles).
- Ridge selecionado: 2; ESS tratado/controle: 202.99/350.80.
- Após calibração, máximo |SMD|: 1.452e-10; máximo KS: 0.09702; razão de variância: 0.833–1.416.
- Efeito placebo agudo em `minutes_share`: 0.0663 (IC95% 0.0164 a 0.1162; p=0.0092).
- Efeito placebo consolidado: 0.0000 (IC95% -0.0551 a 0.0551; p=0.9991).

## Limites

Esta primeira falsificação cobre `minutes_share`, não npxG+xA, lesões ou dados
físicos: os snapshots locais de Understat e lesões para 2021/22 não estavam
presentes. Os pesos foram recalculados com covariáveis pré-pausa disponíveis
(idade, valor, utilização, titularidade, força de clube/adversário e trajetória
de minutos), sem reutilizar pesos de 2022/23. A elegibilidade nacional prioriza
a tabela auditada de 2022 quando o atleta está presente nela; para demais atletas,
usa cidadania primária do Transfermarkt e deve ser lida como proxy.

Não há holdout pré-pausa homogêneo nesta temporada: em 07/11/2021, Premier
League e Bundesliga haviam disputado somente 11 rodadas. Assim, todos os oito
jogos pré-pausa disponíveis para a especificação foram usados na calibração; o
placebo não acrescenta um teste independente de pré-tendência. Além disso, a
coorte observável em 2021/22 (267 futuros convocados) não é idêntica à de
2022/23 (314 convocados), e o snapshot público baixado em 13/07/2026 pode
diferir do snapshot histórico original usado na análise principal.
