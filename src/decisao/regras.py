"""Base de conhecimento sobre descarte (cores conforme Resolução CONAMA nº 275/2001)."""

CONHECIMENTO = {
    "cardboard": {
        "nome": "papelão", "reciclavel": True, "lixeira": "azul",
        "preparo": "Desmonte e dobre as caixas para ocupar menos espaço; mantenha-as secas. "
                   "Papelão molhado ou sujo de gordura (ex.: caixa de pizza) vai no rejeito.",
    },
    "paper": {
        "nome": "papel", "reciclavel": True, "lixeira": "azul",
        "preparo": "Descarte seco e sem amassar muito. Papel higiênico, guardanapo, "
                   "papel engordurado e papel carbono não são recicláveis.",
    },
    "plastic": {
        "nome": "plástico", "reciclavel": True, "lixeira": "vermelha",
        "preparo": "Retire restos de comida e dê uma enxaguada rápida. Garrafas PET podem "
                   "ser amassadas e tampadas para reduzir volume.",
    },
    "metal": {
        "nome": "metal", "reciclavel": True, "lixeira": "amarela",
        "preparo": "Enxágue latas e amasse para reduzir volume. Latas de tinta, aerossóis e "
                   "pilhas exigem descarte especial — não vão na coleta comum.",
    },
    "glass": {
        "nome": "vidro", "reciclavel": True, "lixeira": "verde",
        "preparo": "Enxágue e retire tampas. Vidro quebrado deve ser embrulhado em jornal ou "
                   "colocado em caixa identificada para proteger os coletores. Espelhos, "
                   "lâmpadas e vidro temperado não entram na reciclagem comum.",
    },
    "trash": {
        "nome": "rejeito (não reciclável)", "reciclavel": False, "lixeira": "cinza",
        "preparo": "Descarte em saco fechado na coleta domiciliar comum. Exemplos: "
                   "guardanapos, papel higiênico, fraldas, embalagens metalizadas e bitucas.",
    },
}

EMOJI_COR = {"azul": "🔵", "vermelha": "🔴", "amarela": "🟡", "verde": "🟢", "cinza": "⚪"}
