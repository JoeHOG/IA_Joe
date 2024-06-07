# Código de Phaser para el Juego con 3 Balas

## Constantes y Variables Globales

### Constantes
- `w`: Ancho del juego (800).
- `h`: Alto del juego (400).

### Variables Globales
- Objetos del juego: `jugador`, `fondo`, `bala`, `nave`, `bala2`, `nave2`, `bala3`, `nave3`.
- Indicadores de disparo: `balaD`, `balaD2`, `balaD3`.
- Controles: `salto`, `avanza`.
- UI: `menu`.
- Velocidades y desplazamientos de las balas: `velocidadBala`, `despBala`, `velocidadBala2`, `despBala2`, `velocidadBala3`, `despBalaHorizontal3`, `despBalaVertical3`, `despBala3`.
- Estado del jugador: `estatusAire`, `estatusSuelo`, `estatusDerecha`, `estatusIzquierda`.
- Redes neuronales: `nnNetwork`, `nnEntrenamiento`, `nnSalida`, `datosEntrenamiento` (array vacío).
- Modo automático: `modoAuto` (booleano), `eCompleto` (booleano).

## Inicialización del Juego
- `var juego = new Phaser.Game(w, h, Phaser.CANVAS, '', { preload: preload, create: create, update: update, render: render });`: Crea una instancia del juego con las dimensiones especificadas y configura las funciones de `preload`, `create`, `update`, y `render`.

## Precarga de Assets
- `preload()`: Carga las imágenes y sprites necesarios para el juego:
  ```javascript
  juego.load.image('fondo', 'assets/game/fondo.jpg');
  juego.load.spritesheet('mono', 'assets/sprites/altair.png', 32, 48);
  juego.load.image('nave', 'assets/game/ufo.png');
  juego.load.image('bala', 'assets/sprites/purple_ball.png');
  juego.load.image('menu', 'assets/game/menu.png');
