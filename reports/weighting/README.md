# Fase 5 — pesos, suporte e pré-tendências

Implementação concluída em 12/07/2026 para a população de risco qualificada:
1.015 jogadores, sendo 314 tratados e 701 controles.

## Especificação selecionada

- propensity score estimado separadamente por estrato liga-posição;
- overlap weights calibrados em todas as covariáveis basais contínuas, nos
  estratos e nos valores jogo a jogo de `minutes_share` e
  `npxg_xa_per_club_match` entre os jogos -8 e -1;
- ridge selecionado automaticamente: 1,5;
- janela holdout de pré-tendências: jogos -14 a -9, referência -8;
- nenhuma variável ou resultado pós-Copa usado para escolher os pesos.

## Diagnósticos principais

| Diagnóstico | Resultado |
|---|---:|
| SMD absoluto máximo após calibração | 2,03e-12 |
| ESS tratados | 201,66 de 314 (64,22%) |
| ESS controles | 354,80 de 701 (50,61%) |
| Tratados dentro do suporte comum | 93,95% |
| Maior peso normalizado | 6,06 |
| QA técnico | 18/18 aprovado |

O teste conjunto de `minutes_share` no holdout não rejeita igualdade
(`p` simulado = 0,8867), mas os intervalos de confiança não ficam inteiramente
dentro da margem de equivalência de ±0,10 desvio-padrão. O maior efeito pontual
é 0,1024 DP. Assim, os pesos estão liberados para a estimação dos modelos, mas
o gate de linguagem causal forte permanece fechado.

O entropy balancing para ATT convergiu, porém é instável (ESS de controles de
147,44; 21,03% e maior peso normalizado de 23,38). Deve aparecer apenas como
análise de sensibilidade.

## Arquivos

- `../../data/processed/analysis_weights.csv`: pesos por jogador;
- `candidate_weighting_diagnostics.csv`: grade e seleção do ridge;
- `balance_diagnostics.csv`: SMD, razão de variâncias e KS;
- `weight_distribution.csv`: distribuição e ESS dos pesos;
- `pretrend_coefficients.csv` e `pretrend_joint_tests.csv`: calibração e holdout;
- `phase5_gate_summary.csv` e `phase5_qa.csv`: decisões e auditoria;
- `weighting_manifest.csv`: hashes, tamanhos e linhagem;
- `love_plot.svg`, `propensity_overlap.svg` e
  `pretrend_minutes_share.svg`: figuras diagnósticas.

## Regra para a Fase 6

Usar exclusivamente `analysis_weight` da base congelada. Como a equivalência no
holdout não foi demonstrada, reportar estimativas com cautela e incluir, antes
de qualquer interpretação causal forte, event study completo, tendências
lineares diferenciadas, placebos temporais e análise de sensibilidade a violações
de tendências paralelas.
