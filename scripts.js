
let net;
let webcam;

const classifier = knnClassifier.create();

const imgE1 = document.getElementById("img");
const descE1 = document.getElementById("descripcion_imagen");
const webCam = document.getElementById("webcam");

// Letras que podrá reconocer el clasificador.
// El último índice se reserva para reconocer que no hay una seña.
const clases = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N",
    "Ñ", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "SIN_SEÑA"
];

// Variables para controlar la estabilidad
let letraCandidata = null;
let contadorEstabilidad = 0;
let ultimaLetraAgregada = null;

let palabraFormada = "";

const FRAMES_NECESARIOS = 8;
const CONFIANZA_MINIMA = 0.75;

// Iniciar aplicación
async function app() {
    try {
        document.getElementById("estadoReconocimiento").textContent =
            "Cargando modelo...";

        // Cargar MobileNet
        net = await mobilenet.load();

        // Clasificar imagen estática
        await displayImagePrediction();

        // Inicializar cámara
        webcam = await tf.data.webcam(webCam);

        document.getElementById("estadoReconocimiento").textContent =
            "Cámara lista. Esperando una seña...";

        // Iniciar reconocimiento continuo
        while (true) {
            const img = await webcam.capture();

            try {
                // Predicción MobileNet
                const result = await net.classify(img);

                document.getElementById("console").innerHTML =
                    "<strong>Predicción:</strong> " +
                    result[0].className +
                    "<br><strong>Probabilidad:</strong> " +
                    (result[0].probability * 100).toFixed(2) + "%";

                // Verificar si existen ejemplos entrenados
                if (classifier.getNumClasses() === 0) {
                    document.getElementById("letraActual").textContent =
                        "El clasificador todavía no tiene ejemplos entrenados.";

                    document.getElementById("estadoReconocimiento").textContent =
                        "Falta preparar el entrenamiento de las señas.";

                    await tf.nextFrame();
                    continue;
                }

                // Extraer características de la imagen
                const activation = net.infer(img, true);

                try {
                    // Clasificación personalizada
                    const prediction =
                        await classifier.predictClass(activation);

                    const indice = Number(prediction.label);
                    const letra = clases[indice];
                    const confianza = prediction.confidences[indice] || 0;

                    reconocerLetra(letra, confianza);

                } finally {
                    activation.dispose();
                }

            } finally {
                img.dispose();
            }

            await tf.nextFrame();
        }

    } catch (error) {
        console.error("Error general:", error);

        document.getElementById("estadoReconocimiento").textContent =
            "Error al iniciar. Revisa los permisos de la cámara y la consola.";
    }
}

// Comprobar que la predicción sea estable
function reconocerLetra(letra, confianza) {

    if (confianza < CONFIANZA_MINIMA) {
        letraCandidata = null;
        contadorEstabilidad = 0;

        document.getElementById("letraActual").textContent =
            "Seña no identificada con suficiente confianza.";

        return;
    }

    // Si no hay una seña, permitir reconocer nuevamente
    if (letra === "SIN_SEÑA") {
        letraCandidata = null;
        contadorEstabilidad = 0;
        ultimaLetraAgregada = null;

        document.getElementById("letraActual").textContent =
            "Esperando una seña...";

        return;
    }

    // Contar las predicciones consecutivas de la misma letra
    if (letra === letraCandidata) {
        contadorEstabilidad++;
    } else {
        letraCandidata = letra;
        contadorEstabilidad = 1;
    }

    document.getElementById("letraActual").textContent =
        "Detectando: " + letra +
        " (" + contadorEstabilidad + "/" + FRAMES_NECESARIOS + ")";

    // Confirmar cuando la predicción sea estable
    if (
        contadorEstabilidad >= FRAMES_NECESARIOS &&
        letra !== ultimaLetraAgregada
    ) {
        agregarLetra(letra);
        ultimaLetraAgregada = letra;
    }
}

// Agregar letra a la palabra
function agregarLetra(letra) {
    palabraFormada += letra;

    document.getElementById("palabra").textContent =
        palabraFormada;

    document.getElementById("letraActual").textContent =
        "Letra reconocida: " + letra;
}

// Borrar palabra
function borrarPalabra() {
    palabraFormada = "";

    document.getElementById("palabra").textContent = "_";
    document.getElementById("letraActual").textContent =
        "Esperando una seña...";

    ultimaLetraAgregada = null;
    letraCandidata = null;
    contadorEstabilidad = 0;
}

// Clasificar imagen estática
async function displayImagePrediction() {
    try {
        const result = await net.classify(imgE1);

        descE1.innerHTML =
            "<pre>" + JSON.stringify(result, null, 2) + "</pre>";

    } catch (error) {
        console.error("Error al clasificar imagen:", error);
    }
}

// Cambiar imagen
let count = 0;

function cambiarImagen() {
    count++;

    imgE1.crossOrigin = "anonymous";
    imgE1.src = "https://picsum.photos/200/300?random=" + count;
}

// Ejecutar aplicación
app();
