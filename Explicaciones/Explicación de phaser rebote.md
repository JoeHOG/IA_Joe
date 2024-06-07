## El código de phaser de las 3 balas se divide en # Código de Phaser para el Juego de las 3 Balas

## Constantes y Variables Globales

### Constantes
- Define las dimensiones del juego, la velocidad del jugador y la bola, los fotogramas por segundo (FPS), y los estilos de texto para la pausa.

### Variables Globales
- Incluyen objetos del juego (jugador, fondo, bola, menú), música de fondo, controladores de entrada (teclas de flecha), y variables para el modo automático y datos de entrenamiento.

## Inicialización del Juego
- `var game = new Phaser.Game(...)`: Crea una instancia del juego con las dimensiones especificadas y configura las funciones de `preload`, `create`, `update`, y `render`.

## Precarga de Assets
- `preload()`: Carga las imágenes, sprites y la música de fondo necesarios para el juego.

## Creación de Elementos del Juego
- `create()`: Inicializa el sistema de física, configura el fondo, crea el sprite del jugador y la bola, y configura la etiqueta de pausa y la música de fondo.

### **Adición del Sonido del Juego**
- Donde se incluye el uso de la musica de fondo del metal slug
- **Precarga de la música de fondo**:
  -Y en la cual se carga con el siguiente comando
  game.load.audio('backgroundMusic', 'assets/sprites/sl.mp3');
