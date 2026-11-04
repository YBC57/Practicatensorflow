
let net;
let webcam;
let ultimaPrediccion = null;
let palabra = "";

const clases = ["A", "B", "C"];

const webCam = document.getElementById("webcam");
const classifier = knnClassifier.create();

async function app() {
    try {
        document.getElementById("estado").textContent =
            "Cargando MobileNet...";

        net = await mobilenet.load();

        document.getElementById("estado").textContent =
            "Activando cámara...";

        webcam = await tf.data.webcam(webCam);

        document.getElementById("estado").textContent =
            "Cámara lista. Agrega ejemplos para entrenar.";

        while (true) {
            const img = await webcam.capture();

            try {
                const result = await net.classify(img);

                document.getElementById("console").innerHTML =
                    "<strong>Predicción:</strong> " +
                    result[0].className +
                    "<br><strong>Probabilidad:</strong> " +
                    (result[0].probability * 100).toFixed(2) + "%";

                if (classifier.getNumClasses() > 0) {
                    const activation = net.infer(img, true);

                    try {
                        const prediction =
                            await classifier.predictClass(activation);

                        const letra = clases[
                            Number(prediction.label)
                        ];

                        const confianza =
                            prediction.confidences[prediction.label] || 0;

                        ultimaPrediccion = {
                            letra: letra,
                            confianza: confianza
                        };

                        document.getElementById("console2").innerHTML =
                            "<strong>Seña:</strong> " + letra +
                            "<br><strong>Confianza:</strong> " +
                            (confianza * 100).toFixed(2) + "%";

                    } finally {
                        activation.dispose();
                    }
                } else {
                    document.getElementById("console2").textContent =
                        "Agrega ejemplos de las señas A, B y C.";
                }
            } finally {
                img.dispose();
            }

            await tf.nextFrame();
        }
    } catch (error) {
        console.error("Error general:", error);

        document.getElementById("estado").textContent =
            "Error: " + error.message;
    }
}

async function addExample(classId) {
    if (!net || !webcam) {
        document.getElementById("estado").textContent =
            "Espera a que el modelo y la cámara estén listos.";
        return;
    }

    const img = await webcam.capture();

    try {
        const activation = net.infer(img, true);

        try {
            classifier.addExample(activation, classId);
        } finally {
            activation.dispose();
        }
    } finally {
        img.dispose();
    }

    actualizarEjemplos();
}

function actualizarEjemplos() {
    const contenedor = document.getElementById("ejemplos");

    contenedor.innerHTML = clases.map((letra, id) => {
        const cantidad = classifier.getClassExampleCount(id);

        return `<p>Seña ${letra}: ${cantidad} ejemplos</p>`;
    }).join("");
}

function agregarLetra() {
    if (!ultimaPrediccion) {
        alert("Primero debes entrenar el clasificador.");
        return;
    }

    if (ultimaPrediccion.confianza < 0.7) {
        alert("La confianza es baja. Intenta mostrar mejor la seña.");
        return;
    }

    palabra += ultimaPrediccion.letra;

    document.getElementById("palabra").textContent = palabra;
}

function borrarUltima() {
    palabra = palabra.slice(0, -1);
    document.getElementById("palabra").textContent = palabra;
}

function limpiarPalabra() {
    palabra = "";
    document.getElementById("palabra").textContent = "";
}

app();
