"""Interface Streamlit: foto do resíduo + pergunta em linguagem natural -> resposta do agente.

Uso:
    streamlit run app.py
"""
import json

import streamlit as st
from PIL import Image

from src.config import CLASSES_PT, JSON_MAPA, MODELO_CNN, MODELO_INTENCOES
from src.decisao.agente import AgenteDescarte
from src.decisao.grafo import MapaColeta
from src.pln.entidades import extrair

st.set_page_config(page_title="Descarte Inteligente — Maceió", page_icon="♻️", layout="wide")


@st.cache_resource
def carregar_agente():
    from src.pln.intencoes import ClassificadorIntencoes
    from src.visao_computacional.classificador import ClassificadorResiduos
    faltando = [p.name for p in (MODELO_CNN, MODELO_INTENCOES) if not p.exists()]
    if faltando:
        return None, faltando
    return AgenteDescarte(ClassificadorResiduos(), ClassificadorIntencoes(), MapaColeta()), []


agente, faltando = carregar_agente()
st.title("♻️ Descarte Inteligente — Maceió")
st.caption("Visão Computacional (CNN) + PLN (intenções e entidades) + decisão (regras e A*)")

if faltando:
    st.error(f"Modelos não encontrados: {', '.join(faltando)}. Rode os treinos descritos no README.")
    st.stop()

bairros = sorted(json.loads(JSON_MAPA.read_text(encoding="utf-8"))["bairros"])
col_img, col_chat = st.columns([1, 1.3])

with col_img:
    arquivo = st.file_uploader("Foto do resíduo", type=["jpg", "jpeg", "png"])
    camera = st.camera_input("…ou tire uma foto agora")
    origem = camera or arquivo
    sig_imagem = getattr(origem, "file_id", None)
    if sig_imagem != st.session_state.get("sig_imagem"):
        st.session_state.sig_imagem = sig_imagem
        st.session_state.material_lembrado = None
        st.session_state.aguardando_bairro = False
    imagem = Image.open(origem) if origem else None
    if imagem is not None:
        st.image(imagem)
        rv = agente.visao.classificar(imagem)
        st.subheader("Visão Computacional")
        st.bar_chart({CLASSES_PT[c]: p for c, p in rv.probabilidades.items()}, horizontal=True)
    bairro = st.selectbox("Seu bairro (opcional)", ["—"] + bairros)
    bairro = None if bairro == "—" else bairro

with col_chat:
    if "historico" not in st.session_state:
        st.session_state.historico = []
    for papel, msg in st.session_state.historico:
        st.chat_message(papel).markdown(msg)

    pergunta = st.chat_input("Ex.: isso é reciclável? / onde levo isso?")
    if pergunta:
        st.chat_message("user").markdown(pergunta)
        ent_atual = extrair(pergunta)
        # Se a mensagem anterior pediu o bairro e esta só traz a localização (sem um verbo
        # claro de "onde descartar"), retoma o mesmo pedido em vez de reclassificar do zero.
        intencao_forcada = ("onde_descartar"
                             if st.session_state.get("aguardando_bairro") and (ent_atual.bairro or bairro)
                             else None)
        resp = agente.responder(pergunta, imagem, bairro,
                                 material_lembrado=st.session_state.get("material_lembrado"),
                                 intencao_forcada=intencao_forcada)
        if ent_atual.material:
            st.session_state.material_lembrado = ent_atual.material
        st.session_state.aguardando_bairro = resp.aguardando_bairro
        st.chat_message("assistant").markdown(resp.texto)
        st.session_state.historico += [("user", pergunta), ("assistant", resp.texto)]
        with st.expander("Como o agente decidiu"):
            r_int = agente.pln.prever(pergunta)
            st.write(f"**Intenção:** {r_int.intencao} ({r_int.confianca:.0%})")
            st.write(f"**Material:** {resp.material} (fonte: {resp.fonte_material})")
            st.write("**Regras disparadas:**")
            for r in resp.regras:
                st.write(f"- {r}")
            if resp.rota:
                st.write("**Rota (A\\*):** " + " → ".join(resp.rota))
