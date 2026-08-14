# Fase 6 — análise confirmatória

Implementação concluída em 12/07/2026 usando os pesos congelados da Fase 5.
A amostra contém 1.015 jogadores, 314 tratados e 701 controles.

## Modelo principal

O modelo inclui efeitos fixos de jogador e clube-partida, overlap weights e
erros-padrão agrupados por jogador. O cluster duplo jogador+clube-partida é
reportado como robustez. A janela pré contém os jogos -8 a -1; a janela aguda,
1 a 4; e a consolidada, 5 a 8.

| Outcome | Efeito agudo | IC 95% | p | Holm |
|---|---:|---:|---:|---:|
| `minutes_share` | 0,0233 | [-0,0260; 0,0726] | 0,355 | 1,000 |
| `in_match_squad` | 0,0306 | [-0,0184; 0,0796] | 0,221 | 1,000 |
| `played` | 0,0144 | [-0,0354; 0,0641] | 0,571 | 1,000 |
| `started` | 0,0095 | [-0,0466; 0,0655] | 0,741 | 1,000 |
| `minutes_played` | 2,096 | [-2,343; 6,535] | 0,355 | 1,000 |
| `npxg_xa_per_club_match` | 0,0127 | [-0,0146; 0,0400] | 0,361 | 1,000 |

Para o outcome primário, o efeito agudo equivale a aproximadamente 2,10 minutos
por partida, 0,060 DP ou 3,39% do baseline dos tratados. O intervalo inclui
efeitos negativos pequenos e efeitos positivos acima do SESOI de 0,05; portanto,
o resultado não comprova ausência de efeito relevante.

## Janela consolidada e robustez

No modelo principal, `minutes_share` aumenta 0,0835 na janela consolidada
(IC 95% 0,0301 a 0,1369; p = 0,0022). A estimativa cai para 0,0509 com efeitos
fixos externos (p = 0,0757) e para 0,0213 ao permitir tendência diferenciada
(p = 0,6973). O PPML encontra razão de taxas de 1,163 para minutos na janela
consolidada, mas não resolve a sensibilidade à tendência. Esse achado deve ser
tratado como sinal não robusto, e não como efeito causal estabelecido.

O placebo temporal usa somente os jogos -14 a -9, que não participaram da
calibração. Para `minutes_share`, a estimativa é -0,0170 (p = 0,494). O teste
ajuda, mas não compensa a falta de equivalência demonstrada na Fase 5.

O DiD duplamente robusto estima ATT, não o ATO principal, e aparece apenas como
verificação de estimando alternativo. Para a janela aguda de `minutes_share`, a
estimativa é 0,0388 (p = 0,254).

## Interpretação congelada

- modelos aptos para reporte: sim;
- efeito agudo estatisticamente detectado: não;
- evidência de ausência de efeito relevante: não;
- aumento consolidado robusto: não;
- linguagem causal forte: não;
- formulação permitida: associação longitudinal ajustada na população de overlap.

## Arquivos

- `confirmatory_coefficients.csv`: modelos principal, externo e com tendência;
- `event_study_coefficients.csv`: coeficientes e bandas simultâneas;
- `holdout_placebo.csv`: falsificação temporal pré-Copa;
- `ppml_robustness.csv`: PPML para minutos e produção ofensiva;
- `dr_att_did.csv`: verificação DiD-ATT de estimando alternativo;
- `cluster_robustness.csv`: multiplicadores de escore por clube e seleção;
- `phase6_gate_summary.csv` e `phase6_qa.csv`: decisões e QA;
- `confirmatory_manifest.csv`: hashes e linhagem;
- `event_study_minutes_share.svg`, `event_study_squad.svg` e
  `forest_acute_outcomes.svg`: figuras.

## Continuidade

As sensibilidades subsequentes foram implementadas na Fase 7 e estão em `reports/robustness/`; a falsificação sazonal de 2021/22 está em `reports/placebo_2021_22/`. A interpretação final permanece associativa, conforme o manuscrito.
