# Pintor de Craques — montagem das imagens

Gera `img/pintor/` a partir dos desenhos de linha (pacote do Gemini):

```
pacote/craques/{jogador}.jpeg   cabeça de frente, com pescoço
pacote/corpos/{pose}.jpeg       corpo sem cabeça (parado, chutando, comemorando, goleiro)
pacote/objetos/{objeto}.jpeg    trofeu, bola, medalha
```

```
pip install numpy pillow scipy
python3 tools/pintor/export.py pacote saida
```

O script encaixa cada cabeça em cada corpo (`{jogador}_{pose}.png`), gera as
miniaturas (`rosto_{jogador}.png`) e os objetos (`obj_{objeto}.png`), e imprime o
`POSE_META` (tamanho e ponto do peito) que vai no `pintor.html`. Depois de gerar,
reduza para 16 tons de cinza (ex.: `Image.quantize(16)`) para deixar os arquivos leves.

Para um jogador novo basta adicionar a cabeça em `craques/` e o item em `PLAYERS`.
As coordenadas da gola de cada pose (`CX`, `VTIP`) estão fixas em `prep.py`.
