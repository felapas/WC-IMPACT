# Tabelas intermediárias

Estas tabelas são recriadas por `scripts/etl/build_interim.py`. Elas não devem
ser editadas manualmente.

| Arquivo | Unidade de observação | Finalidade |
|---|---|---|
| `fifa_squads_2022.csv` | jogador convocado | Tratamento oficial, clube, posição e nascimento |
| `big5_player_match_2022_23.csv` | jogador na súmula de um jogo de clube | Minutos, titularidade, banco, produção e contexto da partida |
| `wc_player_match_2022.csv` | jogador na súmula de um jogo da Copa | Dose e métricas StatsBomb por partida |
| `wc_player_tournament_2022.csv` | jogador na Copa | Dose e qualidade agregadas no torneio |
| `understat_player_match_2022_23.csv` | jogador com finalização ou passe para finalização | xG, npxG, xA e volume ofensivo por partida |
| `understat_transfermarkt_game_crosswalk.csv` | jogo | Ponte Understat para Transfermarkt |
| `understat_transfermarkt_player_crosswalk.csv` | jogador por liga | Ponte de IDs Understat para Transfermarkt |
| `fifa_player_crosswalk.csv` | convocado | Ponte FIFA, StatsBomb e Transfermarkt |
| `linkage_review.csv` | ligação incerta | Fila explícita para revisão manual; nunca forçar automaticamente |
| `qa_summary.csv` | teste de qualidade | Cobertura, unicidade e taxas mínimas de linkagem |

Os dados de clube representam jogadores presentes na súmula. Ausências totais
por lesão ou não convocação serão adicionadas ao painel analítico na fase
seguinte, usando elenco, transferências e histórico de lesões.
