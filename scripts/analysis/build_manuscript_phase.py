"""Fase 8: tabelas, figuras, manuscrito Markdown e DOCX revisável."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from textwrap import wrap

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "manuscript"
TABLES = REPORT / "tables"
FIGURES = REPORT / "figures"
DOCX_PATH = REPORT / "MANUSCRITO_FAME_2026.docx"
MD_PATH = REPORT / "MANUSCRITO_FAME_2026.md"
CHECKLIST_PATH = REPORT / "SUBMISSION_CHECKLIST.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "flow": pd.read_csv(ROOT / "reports" / "design" / "sample_flow.csv"),
        "balance": pd.read_csv(ROOT / "reports" / "weighting" / "balance_diagnostics.csv"),
        "weights": pd.read_csv(ROOT / "reports" / "weighting" / "candidate_weighting_diagnostics.csv"),
        "weight_dist": pd.read_csv(ROOT / "reports" / "weighting" / "weight_distribution.csv"),
        "confirm": pd.read_csv(ROOT / "reports" / "confirmatory" / "confirmatory_coefficients.csv"),
        "events": pd.read_csv(ROOT / "reports" / "confirmatory" / "event_study_coefficients.csv"),
        "placebo": pd.read_csv(ROOT / "reports" / "confirmatory" / "holdout_placebo.csv"),
        "windows": pd.read_csv(ROOT / "reports" / "robustness" / "window_sensitivity.csv"),
        "specs": pd.read_csv(ROOT / "reports" / "robustness" / "sample_weight_sensitivity.csv"),
        "trend": pd.read_csv(ROOT / "reports" / "robustness" / "parallel_trends_sensitivity.csv"),
        "dose": pd.read_csv(ROOT / "reports" / "robustness" / "dose_response_associational.csv"),
        "hetero": pd.read_csv(ROOT / "reports" / "robustness" / "heterogeneity_interactions.csv"),
        "coverage": pd.read_csv(ROOT / "reports" / "robustness" / "coverage_matrix.csv"),
    }


OUTCOME_LABELS = {
    "minutes_share": "Proporção de minutos",
    "in_match_squad": "Na súmula",
    "played": "Jogou",
    "started": "Titular",
    "minutes_played": "Minutos jogados",
    "npxg_xa_per_club_match": "npxG+xA por jogo do clube",
}


def fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}".replace("-0.000", "0.000")


def ci(lower: float, upper: float, digits: int = 3) -> str:
    return f"[{fmt(lower, digits)}; {fmt(upper, digits)}]"


def make_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    flow_labels = {
        "01_baseline_club_assigned":"Clube basal atribuído",
        "02_outfield":"Jogadores de linha",
        "03_minimum_pre_games":"Mínimo de jogos pré-Copa",
        "04_minimum_pre_minutes":"Mínimo de 180 minutos",
        "05_baseline_only_eligible":"Elegibilidade apenas pelo baseline",
        "06_qualified_nation_risk_set":"População de risco qualificada",
        "07_same_nation_any_support":"Suporte da mesma seleção",
    }
    t1 = data["flow"].copy()
    t1["Etapa"] = t1["step"].map(flow_labels)
    t1["Total"] = t1["total"].astype(int)
    t1["Convocados"] = t1["treated"].astype(int)
    t1["Controles"] = t1["controls"].astype(int)
    t1["% inicial"] = (100 * t1["share_of_initial"]).map(lambda x: fmt(x, 1))
    t1 = t1[["Etapa","Total","Convocados","Controles","% inicial"]]

    selected = data["weights"][data["weights"]["selected"]].iloc[0]
    t2 = pd.DataFrame([
        ("Jogadores na população de overlap", 1015),
        ("Convocados", 314),
        ("Controles", 701),
        ("Ridge selecionado", fmt(selected["ridge"], 1)),
        ("SMD absoluto máximo após calibração", f"{selected['max_abs_smd']:.2e}"),
        ("ESS convocados", f"{selected['treated_ess']:.2f} ({100*selected['treated_ess_fraction']:.1f}%)"),
        ("ESS controles", f"{selected['control_ess']:.2f} ({100*selected['control_ess_fraction']:.1f}%)"),
        ("Convocados em suporte comum", f"{100*selected['treated_support_fraction']:.2f}%"),
        ("Maior peso normalizado", fmt(selected["max_normalized_weight"], 2)),
    ], columns=["Diagnóstico","Resultado"])

    primary = data["confirm"][data["confirm"]["specification"].eq("primary_player_club_game_fe")].copy()
    primary = primary[primary["coefficient"].isin(["treated_x_acute","treated_x_consolidated"])]
    window_labels = {"treated_x_acute":"Aguda (jogos 1-4)", "treated_x_consolidated":"Consolidada (jogos 5-8)"}
    t3_rows = []
    for row in primary.itertuples(index=False):
        t3_rows.append({
            "Outcome":OUTCOME_LABELS[row.outcome],
            "Janela":window_labels[row.coefficient],
            "Estimativa":fmt(row.estimate, 3),
            "IC 95%":ci(row.ci_lower_player,row.ci_upper_player,3),
            "p":fmt(row.p_value_player,3),
            "p Holm agudo":fmt(row.p_value_holm_acute_family,3) if row.coefficient=="treated_x_acute" else "-",
        })
    t3 = pd.DataFrame(t3_rows)

    t4_rows = []
    for row in data["windows"][data["windows"]["outcome"].eq("minutes_share")].itertuples(index=False):
        t4_rows.append({"Família":"Janela", "Especificação":f"{int(row.post_end)} jogos", "Estimativa":fmt(row.estimate,3), "IC 95%":ci(row.ci_lower,row.ci_upper,3), "p":fmt(row.p_value,3)})
    for label in ["overlap","matching_1_to_3","unweighted","same_nation_support","baseline_minutes_ge_360","entropy_att"]:
        row = data["specs"][(data["specs"]["specification"].eq(label)) & data["specs"]["coefficient"].eq("treated_x_acute")].iloc[0]
        t4_rows.append({"Família":"Método/amostra", "Especificação":label.replace("_"," "), "Estimativa":fmt(row["estimate"],3), "IC 95%":ci(row["ci_lower"],row["ci_upper"],3), "p":fmt(row["p_value"],3)})
    t4 = pd.DataFrame(t4_rows)

    t5_rows=[]
    for row in data["dose"].itertuples(index=False):
        t5_rows.append({"Família":"Dose", "Termo":row.term.replace("_"," "), "Estimativa":fmt(row.estimate,3), "p":fmt(row.p_value,3), "p BH":fmt(row.p_value_bh,3)})
    for row in data["hetero"].itertuples(index=False):
        t5_rows.append({"Família":"Heterogeneidade", "Termo":row.coefficient.replace("acute_difference_","").replace("_"," "), "Estimativa":fmt(row.estimate,3), "p":fmt(row.p_value,3), "p BH":fmt(row.p_value_bh,3)})
    t5=pd.DataFrame(t5_rows)
    return {"table1_sample_flow":t1,"table2_weighting":t2,"table3_confirmatory":t3,"table4_robustness":t4,"table5_exploratory":t5}


def fonts(size: int, bold: bool=False) -> ImageFont.FreeTypeFont:
    path = Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size)


BLUE=(37,99,235); ORANGE=(234,88,12); NAVY=(15,39,71); GRAY=(100,116,139); LIGHT=(226,232,240); WHITE=(255,255,255); BLACK=(17,24,39); RED=(185,28,28)


def draw_title(draw: ImageDraw.ImageDraw, title: str, subtitle: str="") -> None:
    draw.text((90,55),title,font=fonts(42,True),fill=NAVY)
    if subtitle: draw.text((90,110),subtitle,font=fonts(25),fill=GRAY)


def save_figure_1(flow: pd.DataFrame, path: Path) -> None:
    image=Image.new("RGB",(1800,1050),WHITE); d=ImageDraw.Draw(image); draw_title(d,"Desenho temporal e fluxo da amostra","FIFA World Cup Qatar 2022 como interrupção no meio da temporada")
    y=245; x0=120; unit=88
    d.line((x0,y,x0+16*unit,y),fill=GRAY,width=4)
    for i,label in enumerate(["-8","-4","-1","Copa","1","4","5","8"]):
        xs=[0,4,7,8,9,12,13,16][i]*unit+x0
        d.line((xs,y-16,xs,y+16),fill=NAVY,width=3); d.text((xs-18,y+30),label,font=fonts(22),fill=BLACK)
    d.rectangle((x0,y-75,x0+8*unit,y-32),fill=(219,234,254)); d.text((x0+245,y-70),"Baseline / pré-Copa",font=fonts(25,True),fill=NAVY)
    d.rectangle((x0+8*unit,y-75,x0+9*unit,y-32),fill=(254,215,170)); d.text((x0+8*unit+12,y-70),"Torneio",font=fonts(20,True),fill=(124,45,18))
    d.rectangle((x0+9*unit,y-75,x0+13*unit,y-32),fill=(220,252,231)); d.text((x0+9*unit+95,y-70),"Aguda",font=fonts(24,True),fill=(20,83,45))
    d.rectangle((x0+13*unit,y-75,x0+17*unit,y-32),fill=(237,233,254)); d.text((x0+13*unit+65,y-70),"Consolidada",font=fonts(24,True),fill=(76,29,149))
    labels=[("Clube basal atribuído",2831),("Jogadores de linha",2503),("≥180 min pré-Copa",1411),("Risco qualificado",1015),("Convocados",314),("Controles",701)]
    maxv=max(v for _,v in labels); y0=420
    for i,(label,value) in enumerate(labels):
        yy=y0+i*92; width=int(1250*value/maxv)
        d.rounded_rectangle((430,yy,430+width,yy+52),radius=10,fill=BLUE if i<4 else (16,185,129) if label=="Convocados" else ORANGE)
        d.text((90,yy+10),label,font=fonts(25),fill=BLACK); d.text((445+width,yy+10),f"{value:,}".replace(",","."),font=fonts(25,True),fill=BLACK)
    image.save(path,dpi=(200,200))


def save_figure_2(balance: pd.DataFrame, path: Path) -> None:
    data=balance[balance["variable_type"].eq("continuous")].copy().sort_values("smd_before",key=lambda x:x.abs(),ascending=False)
    image=Image.new("RGB",(1800,1250),WHITE); d=ImageDraw.Draw(image); draw_title(d,"Balanceamento antes e depois da ponderação","Diferenças médias padronizadas absolutas")
    left,right,top,bottom=520,1700,180,1130; maximum=max(.9,float(data[["smd_before","smd_after"]].abs().to_numpy().max())*1.05)
    x=lambda v:left+abs(float(v))/maximum*(right-left)
    for threshold,color in [(0.05,GRAY),(0.10,RED)]: d.line((x(threshold),top,x(threshold),bottom),fill=color,width=3)
    label_map={"market_value_log":"valor de mercado (log)","pre_club_ppg":"força do clube","pre_npxg_xa":"npxG+xA pré-Copa","pre_minutes_share":"proporção de minutos","pre_start_rate":"taxa de titularidade","age_at_world_cup":"idade","pre_trend_npxg_xa":"tendência npxG+xA","pre_opponent_ppg":"força dos adversários","pre_injury_days":"dias lesionado","pre_squad_rate":"taxa na súmula","pre_trend_minutes_share":"tendência de minutos"}
    for i,row in enumerate(data.itertuples(index=False)):
        y=top+40+i*78; label=label_map.get(row.variable,row.variable.replace("_"," "))
        d.text((70,y-14),label,font=fonts(23),fill=BLACK)
        d.line((x(row.smd_before),y,x(row.smd_after),y),fill=LIGHT,width=5)
        d.ellipse((x(row.smd_before)-9,y-9,x(row.smd_before)+9,y+9),fill=ORANGE)
        d.rectangle((x(row.smd_after)-8,y-8,x(row.smd_after)+8,y+8),fill=BLUE)
    d.text((left,1170),"SMD absoluto    círculo: antes    quadrado: depois",font=fonts(23),fill=GRAY)
    image.save(path,dpi=(200,200))


def save_figure_3(events: pd.DataFrame, path: Path) -> None:
    data=events[events["outcome"].eq("minutes_share")].sort_values("relative_club_match")
    image=Image.new("RGB",(1800,1050),WHITE); d=ImageDraw.Draw(image); draw_title(d,"Event study da utilização","Efeito em desvios-padrão; bandas simultâneas de 95%")
    left,right,top,bottom=150,1710,180,890; minimum=-.45; maximum=.60
    xs=np.linspace(left,right,len(data)); y=lambda v:top+(maximum-float(v))/(maximum-minimum)*(bottom-top)
    for tick in [-.4,-.2,0,.2,.4,.6]:
        d.line((left,y(tick),right,y(tick)),fill=LIGHT,width=2); d.text((70,y(tick)-12),fmt(tick,1),font=fonts(21),fill=GRAY)
    split=(xs[6]+xs[7])/2; d.line((split,top,split,bottom),fill=GRAY,width=3)
    points=[]
    for xpos,row in zip(xs,data.itertuples(index=False)):
        d.line((xpos,y(row.simultaneous_lower_sd),xpos,y(row.simultaneous_upper_sd)),fill=(148,163,184),width=9)
        d.ellipse((xpos-9,y(row.effect_sd)-9,xpos+9,y(row.effect_sd)+9),fill=BLUE); points.append((xpos,y(row.effect_sd)))
        d.text((xpos-15,bottom+32),str(int(row.relative_club_match)),font=fonts(20),fill=BLACK)
    d.line(points,fill=BLUE,width=4); d.text((650,965),"Jogo relativo (referência: -1)",font=fonts(24),fill=GRAY)
    image.save(path,dpi=(200,200))


def save_figure_4(confirm: pd.DataFrame, path: Path) -> None:
    data=confirm[(confirm["specification"].eq("primary_player_club_game_fe")) & confirm["coefficient"].eq("treated_x_acute")].copy()
    image=Image.new("RGB",(1800,950),WHITE); d=ImageDraw.Draw(image); draw_title(d,"Associações agudas confirmatórias","Estimativas padronizadas e IC 95%; nenhum p de Holm < 0,05")
    left,right,top,bottom=530,1700,180,820; minimum=-.25; maximum=.30
    x=lambda v:left+(float(v)-minimum)/(maximum-minimum)*(right-left)
    for tick in [-.2,-.1,0,.1,.2,.3]:
        d.line((x(tick),top,x(tick),bottom),fill=LIGHT if tick else GRAY,width=2 if tick else 3)
        d.text((x(tick)-20,bottom+24),fmt(tick,1),font=fonts(20),fill=GRAY)
    for i,row in enumerate(data.itertuples(index=False)):
        y=top+55+i*95; sd=row.baseline_weighted_sd; est=row.estimate/sd; lo=row.ci_lower_player/sd; hi=row.ci_upper_player/sd
        d.text((70,y-14),OUTCOME_LABELS[row.outcome],font=fonts(25),fill=BLACK)
        d.line((x(lo),y,x(hi),y),fill=BLUE,width=5); d.ellipse((x(est)-10,y-10,x(est)+10,y+10),fill=BLUE)
    d.text((780,880),"Efeito agudo (DP)",font=fonts(24),fill=GRAY); image.save(path,dpi=(200,200))


def save_figure_5(specs: pd.DataFrame, path: Path) -> None:
    data=specs[specs["coefficient"].eq("treated_x_acute")].copy()
    labels={"overlap":"Overlap principal","entropy_att":"Entropy ATT (instável)","matching_1_to_3":"Matching 1:3","unweighted":"Sem pesos","overlap_truncated_95":"Trunc. 95%","overlap_truncated_99":"Trunc. 99%","same_nation_support":"Same-nation","baseline_minutes_ge_360":"Baseline ≥360 min"}
    image=Image.new("RGB",(1800,1050),WHITE); d=ImageDraw.Draw(image); draw_title(d,"Curva de especificações do efeito agudo","Métodos de peso, truncamentos e amostras")
    left,right,top,bottom=540,1700,175,900; minimum=-.05; maximum=.16; x=lambda v:left+(float(v)-minimum)/(maximum-minimum)*(right-left)
    for tick in [-.05,0,.05,.10,.15]:
        d.line((x(tick),top,x(tick),bottom),fill=LIGHT if tick else GRAY,width=2 if tick else 3)
        d.text((x(tick)-20,bottom+25),fmt(tick,2),font=fonts(20),fill=GRAY)
    for i,row in enumerate(data.itertuples(index=False)):
        y=top+45+i*84; color=ORANGE if row.specification=="entropy_att" else BLUE
        d.text((70,y-13),labels[row.specification],font=fonts(24),fill=BLACK)
        d.line((x(row.ci_lower),y,x(row.ci_upper),y),fill=color,width=5); d.ellipse((x(row.estimate)-9,y-9,x(row.estimate)+9,y+9),fill=color)
    d.text((810,980),"Efeito em minutes share",font=fonts(24),fill=GRAY); image.save(path,dpi=(200,200))


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    header="| "+" | ".join(frame.columns)+" |"
    separator="|"+"|".join(["---"]*len(frame.columns))+"|"
    rows=[]
    for row in frame.astype(str).itertuples(index=False,name=None): rows.append("| "+" | ".join(value.replace("|","/") for value in row)+" |")
    return "\n".join([header,separator]+rows)


TITLE="Da Copa ao retorno ao clube: associação entre convocação para a FIFA World Cup Qatar 2022 e desempenho pós-torneio nas cinco grandes ligas europeias"
ENGLISH_TITLE="From the World Cup back to club football: squad selection for Qatar 2022 and post-tournament performance across Europe’s Big Five leagues"


ABSTRACT=("A Copa do Mundo de 2022 interrompeu as temporadas europeias no meio do calendário e criou uma oportunidade singular para comparar o retorno ao clube de jogadores convocados e não convocados. Construímos um painel jogador-partida das cinco grandes ligas europeias e restringimos a análise a jogadores de linha com pelo menos 180 minutos pré-Copa e elegibilidade para seleções classificadas. A população de risco incluiu 1.015 jogadores (314 convocados e 701 controles). Estimamos propensity scores por liga e posição e aplicamos overlap weights calibrados em covariáveis basais, estratos e histórico jogo a jogo. Os modelos principais incluíram efeitos fixos de jogador e clube-partida, erros agrupados por jogador e janelas aguda (jogos 1-4) e consolidada (5-8). A associação aguda com a proporção de minutos foi 0,0233 (IC95% -0,0260 a 0,0726; p=0,355), equivalente a 2,10 minutos por partida. Nenhum dos seis outcomes agudos permaneceu significativo após correção Holm. A estimativa consolidada de utilização foi positiva no modelo principal, mas diminuiu com efeitos fixos externos e desapareceu ao permitir tendência diferenciada. O placebo pré-Copa não indicou ruptura, porém a equivalência das pré-tendências não foi demonstrada. Sensibilidades a violações de tendências paralelas mostraram que o padrão consolidado não é robusto a desvios pequenos. Concluímos que os dados não sustentam deterioração aguda detectável, mas também não excluem efeitos relevantes; o padrão tardio deve ser interpretado como associação longitudinal ajustada, não como efeito causal estabelecido.")


SECTIONS = [
    ("1. Introdução", [
        "A FIFA World Cup Qatar 2022 foi realizada entre novembro e dezembro, interrompendo as principais ligas europeias no meio da temporada. Diferentemente de Copas disputadas após o encerramento do calendário doméstico, essa configuração reduziu a separação temporal entre o torneio e o retorno ao clube e tornou mais plausível um desenho longitudinal de curto prazo. Ao mesmo tempo, convocação, minutos na Copa e progressão da seleção não foram aleatórios: jogadores convocados eram, em média, mais valiosos, mais utilizados e mais produtivos antes do torneio.",
        "A literatura relaciona congestionamento competitivo a desempenho e lesões. Ekstrand, Waldén e Hägglund observaram que maior exposição imediatamente antes da Copa de 2002 se associou a lesão ou subdesempenho durante o torneio [1]. Revisões posteriores encontraram que calendários congestionados podem elevar a incidência de lesões e afetar ações de alta intensidade ou componentes técnicos, ainda que os resultados sejam heterogêneos [2,3]. Para 2022, Branquinho et al. destacaram a combinação incomum de calendário, condições ambientais e carga acumulada [4].",
        "Estudos recentes da LaLiga compararam oito jogos antes e depois da Copa. Entre não participantes, Reverte-Pagola et al. relataram melhora em diversas métricas de corrida, com redução de velocidade máxima [5]. Entre participantes, Pecci et al. também observaram melhora em vários indicadores de carga externa [6]. Esses trabalhos oferecem medidas físicas de alta qualidade, mas concentram-se em uma liga e usam comparações retrospectivas pré/pós. Permanece aberta a questão de como convocados se comparam a controles elegíveis em várias ligas, após ajuste explícito para utilização, produção e trajetória prévias.",
        "O objetivo deste estudo foi estimar a associação entre convocação para a Copa de 2022 e utilização e produção ofensiva nos oito jogos subsequentes do clube. A hipótese pré-especificada era de deterioração aguda, ou menor melhora, entre convocados. A contribuição metodológica é combinar uma população de risco baseada em elegibilidade nacional, overlap weighting, efeitos fixos de jogador e clube-partida, event study e diagnósticos de sensibilidade. Diante da incerteza sobre tendências paralelas, toda interpretação é deliberadamente associativa.",
    ]),
    ("2. Métodos", []),
    ("2.1 Desenho, fontes e unidade de análise", [
        "Realizamos estudo observacional longitudinal da temporada 2022/23 nas primeiras divisões de Inglaterra, Espanha, Itália, Alemanha e França. A unidade de análise foi jogador-partida do clube. As fontes gratuitas incluíram convocação e relatórios oficiais da FIFA, StatsBomb Open Data para a Copa, transfermarkt-datasets para partidas, aparições, escalações e valores, football-datasets para lesões e Understat para finalizações e produção esperada. Os dados brutos foram preservados de forma imutável e cada etapa produziu manifestos com SHA-256.",
        "A data de corte basal foi 13 de novembro de 2022. A exposição foi presença na lista final oficial de uma seleção classificada. Não houve jogo relativo zero: os jogos -8 a -1 formaram o baseline principal; 1 a 4, a janela aguda; e 5 a 8, a janela consolidada. Jogos -14 a -9 foram reservados como holdout para diagnósticos não utilizados na calibração dos pesos.",
    ]),
    ("2.2 População e outcomes", [
        "Incluímos jogadores de linha com vínculo basal válido, pelo menos quatro jogos de clube e 180 minutos antes da Copa. Para reduzir controles estruturalmente inelegíveis, a população principal foi restrita a jogadores elegíveis para uma das 32 seleções classificadas. Elegibilidade foi construída com cidadania observada e auditoria manual das exceções documentadas. Lesões e transferências posteriores não determinaram inclusão, evitando condicionamento em variáveis pós-exposição.",
        "O outcome primário foi a proporção dos 90 minutos possíveis em cada partida do clube. Outcomes confirmatórios adicionais foram presença na súmula, participação, titularidade, minutos jogados e npxG+xA por partida do clube. Zeros foram mantidos para jogadores ativos fora da súmula ou reservas não utilizados. O menor efeito de interesse substantivo para o outcome primário foi congelado em 0,05, equivalente a 4,5 minutos por partida.",
    ]),
    ("2.3 Ponderação e estimando", [
        "Estimamos propensity scores dentro de estratos liga-posição por regressão logística ridge. A especificação foi escolhida apenas com diagnósticos pré-Copa. Os overlap weights, definidos pela probabilidade estimada de pertencer ao grupo oposto, direcionam o estimando para a população com maior sobreposição observável [7,8]. Em seguida, calibramos os pesos em covariáveis contínuas basais, proporções dos estratos e valores jogo a jogo de utilização e produção entre -8 e -1.",
        "O estimando principal foi o efeito médio na população de overlap (ATO). Entropy balancing para ATT e matching 1:3 foram sensibilidades, não substitutos do principal. Os gates exigiram SMD absoluto máximo de 0,10, 0,05 nas covariáveis centrais, ESS de pelo menos 50% em cada grupo e suporte para pelo menos 80% dos convocados.",
    ]),
    ("2.4 Modelos e inferência", [
        "O modelo principal foi OLS/LPM ponderado com efeitos fixos de jogador e clube-partida e interações convocado×janela aguda e convocado×janela consolidada. Erros-padrão foram agrupados por jogador; cluster duplo jogador+clube-partida e multiplicadores de escore por clube e seleção foram robustezes. Uma especificação externa substituiu clube-partida por liga-posição-semana e incluiu contexto da partida.",
        "Estimamos event study para jogos -8 a -2 e 1 a 8, com -1 como referência, intervalos pontuais e bandas simultâneas. Aplicamos Holm à família dos seis outcomes agudos e Benjamini-Hochberg às análises exploratórias. PPML foi usado para minutos e produção não negativa. Um DiD duplamente robusto para ATT foi reportado como estimando alternativo.",
        "A credibilidade das tendências foi avaliada por teste conjunto, intervalos de equivalência, placebo no holdout e uma análise transparente de violações relativas inspirada por Rambachan e Roth [9]. Essa última não reproduz o pacote HonestDiD oficial e foi rotulada como aproximação. Também variamos janelas, pesos, amostras, história prévia e exclusões leave-one-out.",
    ]),
    ("3. Resultados", []),
    ("3.1 Amostra, suporte e balanceamento", [
        "Dos 2.831 jogadores com clube basal atribuído, 2.503 eram jogadores de linha e 1.411 atingiram 180 minutos pré-Copa. A população de risco qualificada reuniu 1.015 jogadores: 314 convocados e 701 controles. A calibração selecionou ridge 1,5. O SMD absoluto máximo após ponderação foi 2,03×10^-12; o ESS foi 201,66 entre convocados e 354,80 entre controles. A proporção de convocados em suporte comum foi 93,95%.",
        "O balanceamento quase exato decorre da calibração explícita e não elimina confundimento não observado. Entropy balancing convergiu, mas produziu ESS de controles de 21% e peso normalizado máximo de 23,38; por isso, permaneceu sensibilidade instável.",
    ]),
    ("3.2 Resultados confirmatórios", [
        "Na janela aguda, a associação com proporção de minutos foi 0,0233 (IC95% -0,0260 a 0,0726; p=0,355), ou 2,10 minutos por partida. O intervalo correspondeu aproximadamente a -2,34 a +6,54 minutos e incluiu o SESOI de 4,5 minutos. Logo, o resultado não demonstrou deterioração aguda, mas também não estabeleceu equivalência ou ausência de efeito relevante.",
        "As estimativas agudas para presença na súmula, jogar, começar como titular e npxG+xA também foram imprecisas. Nenhum p ajustado por Holm foi inferior a 0,05. O DiD-ATT para utilização aguda foi 0,0388 (p=0,254), coerente em direção, embora direcionado a outro estimando.",
        "Na janela consolidada, o modelo principal estimou 0,0835 em proporção de minutos (IC95% 0,0301 a 0,1369; p=0,002), equivalente a 7,51 minutos. Entretanto, a estimativa caiu para 0,0509 com efeitos fixos externos (p=0,076) e para 0,0213 com tendência diferenciada (p=0,697). O PPML reproduziu aumento consolidado em minutos, mas não resolveu a sensibilidade à tendência.",
    ]),
    ("3.3 Dinâmica, placebos e robustez", [
        "O event study mostrou estimativas prévias próximas de zero em magnitude pontual, mas bandas simultâneas amplas. A janela holdout -14 a -9 não rejeitou igualdade conjunta para utilização, porém os intervalos não ficaram inteiramente dentro da margem de equivalência de ±0,10 DP. O placebo temporal no holdout foi -0,0170 (p=0,494). Esses resultados são compatíveis com ausência de grande ruptura prévia, mas não provam tendências paralelas.",
        "Em janelas simétricas, a estimativa de utilização foi 0,0209 em quatro jogos, 0,0413 em seis e 0,0534 em oito. O crescimento com o horizonte sugere padrão tardio; selecionar oito jogos pela significância seria inadequado. Overlap, matching, ausência de pesos, truncamentos e same-nation produziram estimativas agudas entre 0,023 e 0,039. Entropy ATT estimou 0,088, mas com instabilidade previamente documentada.",
        "O leave-one-league-out variou de -0,0020 a 0,0442 para a janela aguda. Nenhuma liga, clube ou seleção isoladamente determinou o sinal consolidado. Contudo, na análise de violações de tendência, uma restrição suave com M=0,25 já fez o limite inferior consolidado cruzar zero. Portanto, o padrão tardio não passou o gate para interpretação causal.",
    ]),
    ("3.4 Dose e heterogeneidade exploratória", [
        "Entre os 314 convocados, 20 não jogaram na Copa, 56 acumularam 1-90 minutos, 110 acumularam 91-270 e 128 ultrapassaram 270. Nenhum coeficiente de categorias ou spline permaneceu significativo após Benjamini-Hochberg, e não surgiu gradiente monotônico robusto.",
        "Interações de defesa versus ataque e histórico prévio de lesão permaneceram abaixo de 0,05 após BH. Como o efeito médio é impreciso e essas famílias são exploratórias, os resultados devem ser tratados como hipóteses para replicação, não como evidência confirmatória de efeito diferencial.",
    ]),
    ("4. Discussão", [
        "O estudo não encontrou evidência robusta de piora aguda na utilização ou produção ofensiva dos convocados após a Copa de 2022. A estimativa primária foi pequena, positiva e imprecisa. Seu intervalo de confiança, porém, ainda admite ganhos superiores ao SESOI e pequenas perdas; assim, um resultado não significativo não equivale a prova de ausência de efeito.",
        "O aumento consolidado observado no modelo principal merece atenção, mas não uma narrativa causal. Ele aparece em diferentes métodos de peso e exclusões leave-one-out, porém enfraquece com efeitos fixos externos e é eliminado por tendência diferenciada ou pequenas violações calibradas aos leads. Uma interpretação plausível é readaptação gradual, retorno de jogadores de seleções eliminadas, rotação do elenco ou regressão à média; os dados não distinguem esses mecanismos.",
        "Os achados dialogam com estudos da LaLiga que relataram estabilidade ou melhora física após a Copa entre participantes e não participantes [5,6]. Nossa análise acrescenta cinco ligas, um grupo de controle elegível, ponderação e dinâmica jogador-partida, mas mede principalmente utilização e produção ofensiva, não tracking físico. Diferenças entre outcomes físicos e decisões de escalação podem explicar parte da divergência aparente.",
        "A ausência de dose monotônica enfraquece uma explicação simples baseada apenas em minutos de torneio. Dose foi endógena ao papel, à força da seleção e à qualidade do jogador, e somente 20 convocados tiveram zero minuto. As interações exploratórias por posição e lesão exigem replicação e shrinkage antes de qualquer aplicação prática.",
    ]),
    ("4.1 Forças", [
        "As principais forças são o desenho pré-especificado, a separação entre elegibilidade e outcomes pós-Copa, a população de risco baseada em seleções classificadas, o balanceamento auditável, a manutenção de zeros de disponibilidade, efeitos fixos de jogador e clube-partida, bandas simultâneas, multiplicidade e uma matriz de robustez que registra também o que não pôde ser estimado.",
    ]),
    ("4.2 Limitações", [
        "Primeiro, convocação não foi aleatória e confundimento não observado permanece possível mesmo após balanceamento. Segundo, a equivalência das pré-tendências não foi demonstrada, e o padrão consolidado é sensível a pequenas violações. Terceiro, o painel termina no oitavo jogo pós-Copa e não permite janela de dez jogos. Quarto, não há temporada adjacente local para placebo 2021/22 ou 2023/24. Quinto, o limiar de 90 minutos exigiria reconstruir pesos; a análise de 360 minutos foi apenas restrição da amostra existente. Sexto, npxG+xA depende de linkage e proxy de assistência do Understat, com maior incerteza que minutos e escalações. Sétimo, métricas físicas, progressão e pressão de clube não estavam disponíveis de forma homogênea nas cinco ligas.",
        "A análise inspirada por HonestDiD é uma aproximação transparente, não uma implementação oficial dos intervalos uniformemente válidos de Rambachan e Roth [9]. O valor de robustez para confundimento também é aproximado em HDFE com cluster. Essas análises devem orientar a linguagem, não substituir procedimentos formais em uma versão futura.",
    ]),
    ("5. Conclusão", [
        "Entre jogadores elegíveis das cinco grandes ligas europeias, convocação para a Copa de 2022 não se associou a deterioração aguda detectável na utilização ou produção ofensiva do clube. A precisão foi insuficiente para excluir efeitos relevantes. Um aumento tardio de utilização apareceu no modelo principal, mas foi sensível a tendências diferenciadas e a pequenas violações de tendências paralelas. O resultado mais defensável é, portanto, uma associação longitudinal ajustada: sem evidência de prejuízo agudo robusto e sem base suficiente para atribuir causalmente o padrão consolidado à Copa.",
    ]),
]


REFERENCES=[
    "Ekstrand J, Waldén M, Hägglund M. A congested football calendar and the wellbeing of players: correlation between match exposure of European footballers before the World Cup 2002 and their injuries and performances during that World Cup. Br J Sports Med. 2004;38(4):493-497. doi:10.1136/bjsm.2003.009134.",
    "Julian R, Page RM, Harper LD. The effect of fixture congestion on performance during professional male soccer match-play: a systematic critical review with meta-analysis. Sports Med. 2021;51(2):255-273. doi:10.1007/s40279-020-01359-9.",
    "Page RM, Field A, Langley B, Harper LD, Julian R. The effects of fixture congestion on injury in professional male soccer: a systematic review. Sports Med. 2023;53(3):667-685. doi:10.1007/s40279-022-01799-5.",
    "Branquinho L, Forte P, Thomatieli-Santos RV, et al. Perspectives on player performance during FIFA World Cup Qatar 2022: a brief report. Sports. 2023;11(9):174. doi:10.3390/sports11090174.",
    "Reverte-Pagola G, Pecci J, del Ojo-López JJ, López del Campo R, Resta R, Feria-Madueño A. Analyzing the impact of non-participation in the FIFA World Cup Qatar 2022 on LaLiga players' physical performance. Front Sports Act Living. 2024;6:1385267. doi:10.3389/fspor.2024.1385267.",
    "Pecci J, Reverte-Pagola G, del Ojo-López JJ, López del Campo R, Resta Serra R, Feria Madueño A. Impact of the FIFA World Cup Qatar 2022 on LaLiga players' physical performance: unveiling insights into external load patterns. Sports Health. 2025. doi:10.1177/19417381251388123.",
    "Li F, Morgan KL, Zaslavsky AM. Balancing covariates via propensity score weighting. J Am Stat Assoc. 2018;113(521):390-400. doi:10.1080/01621459.2016.1260466.",
    "Li F, Thomas LE, Li F. Addressing extreme propensity scores via the overlap weights. Am J Epidemiol. 2019;188(1):250-257. doi:10.1093/aje/kwy201.",
    "Rambachan A, Roth J. A more credible approach to parallel trends. Rev Econ Stud. 2023;90(5):2555-2591. doi:10.1093/restud/rdad018.",
    "Orhant E, Chapellier JF, Carling C. The impact of a mid-season FIFA World Cup on injury occurrence and patterns in French professional soccer clubs. Res Sports Med. 2024;32(6):1-10. doi:10.1080/15438627.2024.2326517.",
]


def build_markdown(tables: dict[str,pd.DataFrame]) -> str:
    parts=[f"# {TITLE}",f"*{ENGLISH_TITLE}*","","## Resumo",ABSTRACT,"","**Palavras-chave:** futebol; Copa do Mundo; diferenças-em-diferenças; overlap weighting; desempenho; event study."]
    table_index={"3.1 Amostra, suporte e balanceamento":[("Tabela 1. Fluxo da amostra",tables["table1_sample_flow"]),("Tabela 2. Ponderação e suporte",tables["table2_weighting"])],"3.2 Resultados confirmatórios":[("Tabela 3. Resultados confirmatórios",tables["table3_confirmatory"])],"3.3 Dinâmica, placebos e robustez":[("Tabela 4. Robustez selecionada",tables["table4_robustness"])],"3.4 Dose e heterogeneidade exploratória":[("Tabela 5. Resultados exploratórios",tables["table5_exploratory"])]}
    figure_index={"3.1 Amostra, suporte e balanceamento":["![Figura 1. Desenho e fluxo](figures/figure1_design_flow.png)","![Figura 2. Balanceamento](figures/figure2_balance.png)"],"3.2 Resultados confirmatórios":["![Figura 3. Event study](figures/figure3_event_study.png)","![Figura 4. Forest plot agudo](figures/figure4_forest_acute.png)"],"3.3 Dinâmica, placebos e robustez":["![Figura 5. Curva de especificações](figures/figure5_specification_curve.png)"]}
    for heading,paragraphs in SECTIONS:
        main_heading=heading.split()[0].endswith(".")
        parts.extend(["",f"## {heading}" if main_heading else f"### {heading}"])
        parts.extend(paragraphs)
        for caption,frame in table_index.get(heading,[]): parts.extend(["",f"**{caption}**","",dataframe_to_markdown(frame)])
        parts.extend(["",*figure_index.get(heading,[])])
    parts.extend(["","## Disponibilidade de dados e código","O repositório contém scripts versionados, dicionário, QA e manifestos SHA-256. StatsBomb Open Data requer atribuição; a redistribuição de derivados do Understat deve respeitar seus termos. A versão pública deve excluir arquivos cuja licença não permita redistribuição.","","## Aspectos éticos","Foram usados dados secundários públicos sem identificadores sensíveis. A determinação institucional sobre dispensa ou necessidade de revisão ética deve ser obtida antes da submissão; esta versão não declara dispensa.","","## Referências"])
    parts.extend([f"{i}. {reference}" for i,reference in enumerate(REFERENCES,1)])
    parts.extend(["","## Apêndice A. Limitações de cobertura",dataframe_to_markdown(tables["table4_robustness"])])
    return "\n\n".join(parts)+"\n"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr=cell._tc.get_or_add_tcPr(); shd=tc_pr.find(qn("w:shd"))
    if shd is None: shd=OxmlElement("w:shd"); tc_pr.append(shd)
    shd.set(qn("w:fill"),fill)


def set_table_geometry(table, widths: list[int]) -> None:
    table.autofit=False; table.alignment=WD_TABLE_ALIGNMENT.LEFT
    tbl_pr=table._tbl.tblPr
    tbl_w=tbl_pr.find(qn("w:tblW"))
    if tbl_w is None: tbl_w=OxmlElement("w:tblW"); tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:type"),"dxa"); tbl_w.set(qn("w:w"),str(sum(widths)))
    tbl_ind=tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None: tbl_ind=OxmlElement("w:tblInd"); tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:type"),"dxa"); tbl_ind.set(qn("w:w"),"120")
    grid=table._tbl.tblGrid
    for child in list(grid): grid.remove(child)
    for width in widths:
        col=OxmlElement("w:gridCol"); col.set(qn("w:w"),str(width)); grid.append(col)
    for row in table.rows:
        for cell,width in zip(row.cells,widths):
            tc_pr=cell._tc.get_or_add_tcPr(); tc_w=tc_pr.find(qn("w:tcW"))
            if tc_w is None: tc_w=OxmlElement("w:tcW"); tc_pr.append(tc_w)
            tc_w.set(qn("w:type"),"dxa"); tc_w.set(qn("w:w"),str(width)); cell.width=Inches(width/1440)
            cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tc_mar=tc_pr.find(qn("w:tcMar"))
            if tc_mar is None: tc_mar=OxmlElement("w:tcMar"); tc_pr.append(tc_mar)
            for side,value in [("top",80),("bottom",80),("start",120),("end",120)]:
                node=tc_mar.find(qn(f"w:{side}"))
                if node is None: node=OxmlElement(f"w:{side}"); tc_mar.append(node)
                node.set(qn("w:w"),str(value)); node.set(qn("w:type"),"dxa")


def repeat_header(row) -> None:
    tr_pr=row._tr.get_or_add_trPr(); tbl_header=OxmlElement("w:tblHeader"); tbl_header.set(qn("w:val"),"true"); tr_pr.append(tbl_header)


def add_table(doc: Document, frame: pd.DataFrame, widths: list[int], font_size: float=8.5) -> None:
    table=doc.add_table(rows=1,cols=len(frame.columns)); table.style="Table Grid"
    for index,column in enumerate(frame.columns):
        cell=table.rows[0].cells[index]; cell.text=str(column); set_cell_shading(cell,"F4F6F9")
        for run in cell.paragraphs[0].runs: run.bold=True; run.font.size=Pt(font_size)
    repeat_header(table.rows[0])
    for values in frame.itertuples(index=False,name=None):
        cells=table.add_row().cells
        for index,value in enumerate(values):
            cells[index].text=str(value)
            for run in cells[index].paragraphs[0].runs: run.font.size=Pt(font_size)
            cells[index].paragraphs[0].paragraph_format.space_after=Pt(0)
            cells[index].paragraphs[0].alignment=WD_ALIGN_PARAGRAPH.LEFT if index==0 else WD_ALIGN_PARAGRAPH.CENTER
    set_table_geometry(table,widths)
    p=doc.add_paragraph(); p.paragraph_format.space_after=Pt(3)


def set_styles(doc: Document) -> None:
    styles=doc.styles
    normal=styles["Normal"]; normal.font.name="Calibri"; normal._element.rPr.rFonts.set(qn("w:ascii"),"Calibri"); normal._element.rPr.rFonts.set(qn("w:hAnsi"),"Calibri"); normal.font.size=Pt(11)
    normal.paragraph_format.space_before=Pt(0); normal.paragraph_format.space_after=Pt(8); normal.paragraph_format.line_spacing=1.333; normal.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    specs={"Heading 1":(16,"2E74B5",18,10),"Heading 2":(13,"2E74B5",12,6),"Heading 3":(12,"1F4D78",8,4)}
    for name,(size,color,before,after) in specs.items():
        style=styles[name]; style.font.name="Calibri"; style._element.rPr.rFonts.set(qn("w:ascii"),"Calibri"); style._element.rPr.rFonts.set(qn("w:hAnsi"),"Calibri"); style.font.size=Pt(size); style.font.bold=True; style.font.color.rgb=RGBColor.from_string(color); style.paragraph_format.space_before=Pt(before); style.paragraph_format.space_after=Pt(after); style.paragraph_format.keep_with_next=True
    caption=styles["Caption"]; caption.font.name="Calibri"; caption.font.size=Pt(9); caption.font.italic=True; caption.font.color.rgb=RGBColor.from_string("4B5563"); caption.paragraph_format.space_before=Pt(6); caption.paragraph_format.space_after=Pt(4); caption.paragraph_format.keep_with_next=True


def add_page_field(paragraph) -> None:
    paragraph.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    run=paragraph.add_run("Página ")
    fld=OxmlElement("w:fldSimple"); fld.set(qn("w:instr"),"PAGE"); run._r.addnext(fld)


def add_caption(doc: Document, text: str) -> None:
    p=doc.add_paragraph(style="Caption"); p.add_run(text)


def add_figure(doc: Document, path: Path, caption: str, width: float=6.25) -> None:
    add_caption(doc,caption)
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.keep_together=True; p.paragraph_format.space_after=Pt(8)
    run=p.add_run(); shape=run.add_picture(str(path),width=Inches(width))
    shape._inline.docPr.set("descr",caption)
    shape._inline.docPr.set("title",caption)


def build_docx(tables: dict[str,pd.DataFrame], figure_paths: list[Path]) -> None:
    doc=Document(); set_styles(doc)
    section=doc.sections[0]; section.page_width=Inches(8.5); section.page_height=Inches(11); section.top_margin=Inches(1); section.bottom_margin=Inches(1); section.left_margin=Inches(1); section.right_margin=Inches(1); section.header_distance=Inches(.492); section.footer_distance=Inches(.492)
    header=section.header.paragraphs[0]; header.text="FAME 2026 | Manuscrito em desenvolvimento"; header.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    for run in header.runs: run.font.name="Calibri"; run.font.size=Pt(8.5); run.font.color.rgb=RGBColor.from_string("64748B")
    add_page_field(section.footer.paragraphs[0])
    spacer=doc.add_paragraph(); spacer.paragraph_format.space_after=Pt(120)
    kicker=doc.add_paragraph(); kicker.alignment=WD_ALIGN_PARAGRAPH.CENTER; kicker.paragraph_format.space_after=Pt(18); r=kicker.add_run("ESTUDO OBSERVACIONAL | FAME 2026"); r.bold=True; r.font.size=Pt(10); r.font.color.rgb=RGBColor.from_string("7A5A00")
    title=doc.add_paragraph(); title.alignment=WD_ALIGN_PARAGRAPH.CENTER; title.paragraph_format.space_after=Pt(12); r=title.add_run(TITLE); r.bold=True; r.font.name="Calibri"; r.font.size=Pt(26); r.font.color.rgb=RGBColor.from_string("0B2545")
    subtitle=doc.add_paragraph(); subtitle.alignment=WD_ALIGN_PARAGRAPH.CENTER; subtitle.paragraph_format.space_after=Pt(24); r=subtitle.add_run(ENGLISH_TITLE); r.italic=True; r.font.size=Pt(13); r.font.color.rgb=RGBColor.from_string("475569")
    meta=doc.add_paragraph(); meta.alignment=WD_ALIGN_PARAGRAPH.CENTER; meta.paragraph_format.space_after=Pt(6); r=meta.add_run("Versão de trabalho baseada nas Fases 4-7 auditadas | 12 julho 2026"); r.font.size=Pt(10); r.font.color.rgb=RGBColor.from_string("64748B")
    note=doc.add_paragraph(); note.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=note.add_run("Autoria, afiliação, aprovação ética e periódico-alvo devem ser definidos antes da submissão."); r.font.size=Pt(9); r.font.italic=True; r.font.color.rgb=RGBColor.from_string("7C2D12")
    doc.add_page_break()
    doc.add_heading("Resumo",level=1); doc.add_paragraph(ABSTRACT)
    p=doc.add_paragraph(); p.add_run("Palavras-chave: ").bold=True; p.add_run("futebol; Copa do Mundo; diferenças-em-diferenças; overlap weighting; desempenho; event study.")
    doc.add_heading("Abstract",level=1); doc.add_paragraph("The 2022 World Cup interrupted European club seasons mid-calendar, creating a unique setting to compare post-tournament club performance among selected and non-selected players. In a weighted player-match panel of 1,015 eligible outfield players across Europe’s Big Five leagues, the acute association with minutes share was 0.0233 (95% CI -0.0260 to 0.0726; p=0.355), equivalent to 2.10 minutes per match. No acute outcome survived Holm correction. A positive consolidated association was sensitive to external fixed effects and differential trends. Holdout diagnostics did not demonstrate pre-trend equivalence, and modest parallel-trend violations overturned the consolidated result. The evidence therefore supports an adjusted longitudinal association, not a strong causal claim.")
    p=doc.add_paragraph(); p.add_run("Keywords: ").bold=True; p.add_run("football; FIFA World Cup; difference-in-differences; overlap weighting; player performance.")
    table_map={"3.1 Amostra, suporte e balanceamento":[("Tabela 1. Fluxo da amostra.",tables["table1_sample_flow"],[3500,1400,1500,1500,1460],8.2),("Tabela 2. Diagnósticos da ponderação principal.",tables["table2_weighting"],[6000,3360],9)],"3.2 Resultados confirmatórios":[("Tabela 3. Associações confirmatórias ponderadas.",tables["table3_confirmatory"],[2350,1850,1100,1800,900,1360],7.7)],"3.3 Dinâmica, placebos e robustez":[("Tabela 4. Robustez selecionada do outcome primário.",tables["table4_robustness"],[1500,2800,1200,2200,1660],8.1)],"3.4 Dose e heterogeneidade exploratória":[("Tabela 5. Dose e interações exploratórias.",tables["table5_exploratory"],[1600,3900,1300,1200,1360],8.0)]}
    figure_map={"3.1 Amostra, suporte e balanceamento":[(figure_paths[0],"Figura 1. Desenho temporal e fluxo da amostra."),(figure_paths[1],"Figura 2. Balanceamento antes e depois dos overlap weights.")],"3.2 Resultados confirmatórios":[(figure_paths[2],"Figura 3. Event study ponderado da proporção de minutos."),(figure_paths[3],"Figura 4. Forest plot das associações agudas confirmatórias.")],"3.3 Dinâmica, placebos e robustez":[(figure_paths[4],"Figura 5. Curva de especificações do efeito agudo.")]}
    for heading,paragraphs in SECTIONS:
        level=1 if heading.split()[0].endswith(".") else 2
        doc.add_heading(heading,level=level)
        for paragraph in paragraphs: doc.add_paragraph(paragraph)
        for caption,frame,widths,size in table_map.get(heading,[]): add_caption(doc,caption); add_table(doc,frame,widths,size)
        for path,caption in figure_map.get(heading,[]): add_figure(doc,path,caption)
    doc.add_heading("6. Disponibilidade de dados e código",level=1)
    doc.add_paragraph("O repositório contém scripts versionados, dicionário de variáveis, arquivos de QA e manifestos SHA-256. StatsBomb Open Data requer atribuição. A redistribuição de dados ou derivados do Understat e de fontes públicas consultadas deve respeitar os termos vigentes. A versão pública do repositório deve excluir qualquer arquivo cuja licença não permita redistribuição.")
    doc.add_heading("7. Aspectos éticos",level=1)
    doc.add_paragraph("O estudo utiliza dados secundários públicos sobre atividade profissional e não contém identificadores sensíveis. A determinação institucional sobre dispensa ou necessidade de revisão ética deve ser obtida antes da submissão; esta versão de trabalho não declara dispensa.")
    doc.add_heading("Referências",level=1)
    for index,reference in enumerate(REFERENCES,1):
        p=doc.add_paragraph(style="Normal"); p.paragraph_format.left_indent=Inches(.25); p.paragraph_format.first_line_indent=Inches(-.25); p.paragraph_format.space_after=Pt(5); p.add_run(f"{index}. {reference}")
    doc.add_page_break(); doc.add_heading("Apêndice metodológico",level=1)
    doc.add_heading("A1. Limitações de cobertura",level=2)
    doc.add_paragraph("As especificações abaixo foram registradas antes da redação final. Itens indisponíveis não foram preenchidos por extrapolação.")
    coverage=pd.read_csv(ROOT/"reports"/"robustness"/"coverage_matrix.csv")
    coverage=coverage.rename(columns={"domain":"Domínio","specification":"Especificação","status":"Status","reason":"Justificativa"})
    add_table(doc,coverage,[1500,2300,2100,3460],7.5)
    doc.add_heading("A2. Regras de interpretação",level=2)
    doc.add_paragraph("O gate de linguagem causal forte permaneceu fechado nas Fases 5, 6 e 7. Resultados de entropy balancing, matching, dose e heterogeneidade não substituem a especificação principal. Valores de p não foram usados como gate de continuidade; magnitude, precisão, balanceamento e sensibilidade orientaram a interpretação.")
    doc.core_properties.title=TITLE; doc.core_properties.subject="Manuscrito FAME 2026"; doc.core_properties.author="Equipe do estudo FAME 2026"; doc.core_properties.keywords="football, World Cup, overlap weighting, event study"
    doc.save(DOCX_PATH)


def main() -> None:
    REPORT.mkdir(parents=True,exist_ok=True); TABLES.mkdir(parents=True,exist_ok=True); FIGURES.mkdir(parents=True,exist_ok=True)
    data=load_inputs(); tables=make_tables(data)
    for name,frame in tables.items(): frame.to_csv(TABLES/f"{name}.csv",index=False,encoding="utf-8",lineterminator="\n")
    figure_paths=[FIGURES/"figure1_design_flow.png",FIGURES/"figure2_balance.png",FIGURES/"figure3_event_study.png",FIGURES/"figure4_forest_acute.png",FIGURES/"figure5_specification_curve.png"]
    save_figure_1(data["flow"],figure_paths[0]); save_figure_2(data["balance"],figure_paths[1]); save_figure_3(data["events"],figure_paths[2]); save_figure_4(data["confirm"],figure_paths[3]); save_figure_5(data["specs"],figure_paths[4])
    MD_PATH.write_text(build_markdown(tables),encoding="utf-8")
    CHECKLIST_PATH.write_text("""# Checklist antes da submissão\n\n- Definir autoria, ordem, afiliações e autor correspondente.\n- Escolher periódico/congresso e adaptar limite de palavras, resumo e referências.\n- Obter determinação institucional sobre revisão ética ou dispensa.\n- Confirmar os termos de redistribuição de todos os derivados, especialmente Understat.\n- Executar HonestDiD oficial em R/Stata se houver ambiente disponível.\n- Priorizar coleta de temporada adjacente para placebo, se o cronograma permitir.\n- Decidir se as interações exploratórias permanecerão no texto ou apenas no apêndice.\n- Revisar título e abstract em inglês com coautor fluente.\n- Inserir declaração de financiamento, conflitos de interesse e contribuições CRediT.\n- Gerar pacote público apenas após auditoria de licenças e anonimização de caminhos locais.\n""",encoding="utf-8")
    build_docx(tables,figure_paths)
    generated=[*sorted(TABLES.glob("*.csv")),*figure_paths,MD_PATH,CHECKLIST_PATH,DOCX_PATH]
    rows=[{"relative_path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":sha256(p),"generated_by":"scripts/analysis/build_manuscript_phase.py"} for p in generated]
    pd.DataFrame(rows).to_csv(REPORT/"manuscript_manifest.csv",index=False,encoding="utf-8",lineterminator="\n")
    print(f"Fase 8 gerada: {len(tables)} tabelas, {len(figure_paths)} figuras, Markdown e DOCX.")


if __name__=="__main__": main()
