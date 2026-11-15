let net;
let webcam;

const classifier = knnClassifier.create();

const imgE1 = document.getElementById("img");
const descE1 = document.getElementById("descripcion_imagen");
const webCam = document.getElementById("webcam");

const clases = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "SIN_SEÑA"
];

const pistas = {
    A: "Puño cerrado; el pulgar queda junto al índice, no extendido hacia arriba.",
    B: "Dedos juntos y extendidos; el pulgar se dobla sobre la palma.",
    C: "Dedos y pulgar curvos, formando la silueta de una C.",
    D: "Índice arriba; los otros dedos y el pulgar se curvan para tocarse.",
    E: "Dedos curvados hacia la palma y pulgar doblado sobre ellos.",
    F: "Pulgar e índice forman un círculo; los otros tres dedos quedan extendidos.",
    G: "Índice y pulgar extendidos horizontalmente y en paralelo.",
    H: "Índice y medio extendidos juntos hacia un lado.",
    I: "Meñique extendido hacia arriba; los demás dedos se cierran.",
    J: "El meñique traza una J en el aire; requiere movimiento.",
    K: "Índice y medio extendidos; el pulgar toca el medio.",
    L: "Índice arriba y pulgar hacia un lado, formando una L.",
    M: "Pulgar bajo los dedos índice, medio y anular.",
    N: "Pulgar bajo los dedos índice y medio.",
    O: "Dedos y pulgar curvos, tocándose para formar una O.",
    P: "Configuración similar a K, orientada hacia abajo.",
    Q: "Configuración similar a G, orientada hacia abajo.",
    R: "Índice y medio extendidos y cruzados.",
    S: "Puño cerrado con el pulgar cruzado por delante de los dedos.",
    T: "Pulgar asomando entre el índice y el medio, con los demás dedos cerrados.",
    U: "Índice y medio juntos y extendidos hacia arriba.",
    V: "Índice y medio extendidos y separados en forma de V.",
    W: "Índice, medio y anular extendidos y separados.",
    X: "Índice extendido con la articulación flexionada.",
    Y: "Pulgar y meñique extendidos; los otros dedos cerrados.",
    Z: "El índice traza una Z en el aire; requiere movimiento.",
    SIN_SEÑA: "Deja la mano fuera del encuadre para indicar que no hay una seña."
};

const STORAGE_KEY = "senalab-asl-classifier-v2";
const WORDS_STORAGE_KEY = "senalab-saved-words-v1";
let letraCandidata = null;
let contadorEstabilidad = 0;
let ultimaLetraAgregada = null;
let palabraFormada = "";
let palabrasGuardadas = [];

const FRAMES_NECESARIOS = 8;
const CONFIANZA_MINIMA = 0.75;
const VECINOS_KNN = 5;

const ejemplosPorClase = {};

clases.forEach((clase, indice) => {
    if (clase !== "SIN_SEÑA") {
        const tarjeta = document.createElement("button");
        tarjeta.type = "button";
        tarjeta.id = "letra-" + clase;
        tarjeta.className = "letter-tile" +
            (clase === "J" || clase === "Z" ? " is-motion" : "");
        tarjeta.disabled = true;
        tarjeta.addEventListener("click", () => addExample(indice));

        const letra = document.createElement("span");
        letra.textContent = clase;
        tarjeta.appendChild(letra);

        const cantidad = document.createElement("span");
        cantidad.id = "ejemplos-" + clase;
        cantidad.className = "letter-count";
        cantidad.textContent = " (0)";
        tarjeta.appendChild(cantidad);

        tarjeta.title = clase + ": " + pistas[clase] +
            " Pulsa para capturar un ejemplo.";
        tarjeta.setAttribute("aria-label", tarjeta.title);
        document.getElementById("alfabeto").appendChild(tarjeta);
    }
});

imgE1.onload = () => {
    displayImagePrediction();
};

// Iniciar aplicación
async function app() {
    const estado = document.getElementById("estadoReconocimiento");

    try {
        estado.textContent = "Cargando MobileNet...";
        net = await mobilenet.load();

        const conocimientoValido = cargarConocimiento();
        cargarPalabrasGuardadas();
        actualizarProgreso();
        if (imgE1.complete && imgE1.naturalWidth > 0) {
            await displayImagePrediction();
        }

        estado.textContent = "Solicitando permiso de cámara...";
        webcam = await tf.data.webcam(webCam);

        estado.textContent = "Cámara y modelo listos.";
        document.getElementById("capturarSinSena").disabled = false;
        document.querySelectorAll(".letter-tile").forEach(
            boton => boton.disabled = false
        );
        if (!conocimientoValido) {
            document.getElementById("estadoEntrenamiento").textContent =
                "No se pudo recuperar el conocimiento guardado. Captura nuevos ejemplos.";
            document.getElementById("letraActual").textContent =
                "El entrenamiento guardado no se pudo cargar.";
        } else if (classifier.getNumClasses() < 2) {
            document.getElementById("estadoEntrenamiento").textContent =
                "Pulsa una letra y SIN_SEÑA para capturar ejemplos de al menos dos clases.";
            document.getElementById("letraActual").textContent =
                "Enseña al menos dos clases para probar el reconocimiento.";
        } else {
            document.getElementById("estadoEntrenamiento").textContent =
                "Ejemplos recuperados de este navegador. Puedes añadir más muestras.";
            document.getElementById("letraActual").textContent =
                "Clasificador listo. Haz una seña enseñada frente a la cámara.";
        }

        // Reconocimiento continuo
        while (true) {
            const img = await webcam.capture();

            try {
                const result = await net.classify(img);
                document.getElementById("console").innerHTML =
                    "<strong>Predicción:</strong> " +
                    result[0].className +
                    "<br><strong>Probabilidad:</strong> " +
                    (result[0].probability * 100).toFixed(2) + "%";

                if (classifier.getNumClasses() < 2) {
                    document.getElementById("console2").textContent =
                        "Agrega ejemplos de al menos dos clases distintas, por ejemplo una letra y SIN_SEÑA.";

                    await tf.nextFrame();
                    continue;
                }

                const activation = net.infer(img, true);

                try {
                    const prediction =
                        await classifier.predictClass(
                            activation,
                            Math.min(VECINOS_KNN, classifier.getNumExamples())
                        );

                    const indice = Number(prediction.label);
                    const letra = clases[indice];

                    const confianza =
                        prediction.confidences[prediction.label] || 0;

                    document.getElementById("console2").innerHTML =
                        "<strong>Predicción:</strong> " + letra +
                        "<br><strong>Probabilidad KNN:</strong> " +
                        (confianza * 100).toFixed(1) + "%";
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

        if (!window.isSecureContext) {
            estado.textContent =
                "La cámara requiere un contexto seguro. Abre el proyecto desde localhost o HTTPS.";
        } else if (error.name === "NotAllowedError") {
            estado.textContent =
                "No se permitió el acceso a la cámara. Autoriza el permiso para este sitio en el navegador.";
        } else if (error.name === "NotFoundError") {
            estado.textContent =
                "No se encontró una cámara conectada o disponible.";
        } else if (error.name === "NotReadableError") {
            estado.textContent =
                "La cámara está ocupada por otra aplicación o no se puede iniciar.";
        } else {
            estado.textContent =
                "Error al iniciar la cámara o el modelo: " + error.message;
        }
    }
}

async function addExample(classId) {
    if (!net || !webcam) {
        document.getElementById("estadoEntrenamiento").textContent =
            "Espera a que MobileNet y la cámara estén listos.";
        return;
    }

    if (!Number.isInteger(classId) || classId < 0 || classId >= clases.length) {
        throw new RangeError("El identificador de la seña no es válido.");
    }

    let img;
    let activation;
    try {
        img = await webcam.capture();
        activation = net.infer(img, true);
        classifier.addExample(activation, classId);

        const key = String(classId);
        ejemplosPorClase[key] = (ejemplosPorClase[key] || 0) + 1;
        actualizarProgreso();
        guardarConocimiento();

        const clase = clases[classId];
        document.getElementById("pistaLetra").textContent =
            "Captura " + ejemplosPorClase[key] + " de " + clase + ". " +
            pistas[clase] +
            (clase === "J" || clase === "Z"
                ? " La captura es estática; para reconocer el movimiento hace falta un modelo temporal."
                : "");
        document.getElementById("estadoEntrenamiento").textContent =
            "Ejemplo añadido para " + clase + ". " + mensajeGuardado;
    } catch (error) {
        console.error("No se pudo capturar el ejemplo:", error);
        document.getElementById("estadoEntrenamiento").textContent =
            "No se pudo añadir el ejemplo: " + error.message;
    } finally {
        if (activation) {
            activation.dispose();
        }
        if (img) {
            img.dispose();
        }
    }
}

function actualizarProgreso() {
    for (const clase of clases) {
        if (clase === "SIN_SEÑA") {
            continue;
        }

        const tarjeta = document.getElementById("letra-" + clase);
        const cantidad = ejemplosPorClase[String(clases.indexOf(clase))] || 0;
        document.getElementById("ejemplos-" + clase).textContent =
            " (" + cantidad + ")";
        tarjeta.classList.toggle(
            "is-learned",
            cantidad > 0
        );
    }

    const aprendidas = clases
        .slice(0, 26)
        .filter((_, indice) => ejemplosPorClase[String(indice)] > 0)
        .length;
    document.getElementById("contadorClases").textContent =
        aprendidas + " / 26";
    document.getElementById("capturarSinSena").textContent =
        "Capturar ejemplo: sin seña (" +
        (ejemplosPorClase[String(clases.indexOf("SIN_SEÑA"))] || 0) + ")";
}

async function displayImagePrediction() {
    if (!net || !imgE1.complete || imgE1.naturalWidth === 0) {
        return;
    }

    try {
        const result = await net.classify(imgE1);
        const pre = document.createElement("pre");
        pre.textContent = JSON.stringify(result, null, 2);
        descE1.replaceChildren(pre);
    } catch (error) {
        console.error("No se pudo clasificar la imagen estática:", error);
        descE1.textContent = "Error al clasificar la imagen: " + error.message;
    }
}

let count = 0;

function cambiarImagen() {
    count++;
    imgE1.crossOrigin = "anonymous";
    imgE1.src = "https://picsum.photos/200/300?random=" + count;
}

function float32ToBase64(values) {
    const bytes = new Uint8Array(
        values.buffer,
        values.byteOffset,
        values.byteLength
    );
    let binary = "";
    const chunkSize = 0x8000;

    for (let offset = 0; offset < bytes.length; offset += chunkSize) {
        binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
    }

    return btoa(binary);
}

function base64ToFloat32(encoded) {
    const binary = atob(encoded);
    const bytes = Uint8Array.from(binary, character => character.charCodeAt(0));

    if (bytes.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
        throw new Error("Los datos guardados del clasificador están incompletos.");
    }

    return new Float32Array(bytes.buffer);
}

let mensajeGuardado = "Ejemplos guardados en este navegador.";

function serializarDataset() {
    const dataset = classifier.getClassifierDataset();
    return Object.entries(dataset).map(([label, tensor]) => ({
        label,
        shape: tensor.shape,
        values: float32ToBase64(tensor.dataSync())
    }));
}

function guardarConocimiento() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({
            version: 1,
            dataset: serializarDataset()
        }));
        mensajeGuardado = "Ejemplos guardados en este navegador.";
    } catch (error) {
        console.error("No se pudo guardar el clasificador:", error);
        mensajeGuardado =
            "No se pudieron guardar los ejemplos en el navegador; seguirán disponibles mientras esta página permanezca abierta.";
    }
}

function restaurarDatasetDesdeObjeto(conocimiento) {
    const dataset = {};
    const cantidades = {};

    if (
        conocimiento.version !== 1 ||
        !Array.isArray(conocimiento.dataset)
    ) {
        throw new Error("El formato del conocimiento guardado no es válido.");
    }

    for (const ejemplo of conocimiento.dataset) {
        const label = Number(ejemplo.label);
        if (
            !Number.isInteger(label) ||
            label < 0 ||
            label >= clases.length ||
            !Array.isArray(ejemplo.shape) ||
            ejemplo.shape.length !== 2 ||
            !ejemplo.shape.every(Number.isInteger) ||
            ejemplo.shape[0] <= 0 ||
            ejemplo.shape[1] <= 0
        ) {
            throw new Error("Hay una clase o dimensión inválida en los datos guardados.");
        }

        const values = base64ToFloat32(ejemplo.values);
        if (values.length !== ejemplo.shape[0] * ejemplo.shape[1]) {
            throw new Error("La cantidad de datos guardados no coincide con su dimensión.");
        }

        dataset[String(label)] = tf.tensor2d(values, ejemplo.shape);
        cantidades[String(label)] = ejemplo.shape[0];
    }

    classifier.clearAllClasses();
    Object.keys(ejemplosPorClase).forEach(k => delete ejemplosPorClase[k]);
    classifier.setClassifierDataset(dataset);
    Object.assign(ejemplosPorClase, cantidades);
}

function cargarConocimiento() {
    try {
        const contenido = localStorage.getItem(STORAGE_KEY);
        if (!contenido) {
            return true;
        }
        restaurarDatasetDesdeObjeto(JSON.parse(contenido));
        return true;
    } catch (error) {
        console.error("No se pudo recuperar el clasificador guardado:", error);
        document.getElementById("estadoEntrenamiento").textContent =
            "No se pudieron recuperar los ejemplos guardados: " + error.message;
        return false;
    }
}

function exportarModelo() {
    if (classifier.getNumClasses() === 0) {
        document.getElementById("estadoEntrenamiento").textContent =
            "No hay ejemplos entrenados para exportar.";
        return;
    }
    const payload = JSON.stringify({
        version: 1,
        dataset: serializarDataset()
    }, null, 2);
    const blob = new Blob([payload], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const enlace = document.createElement("a");
    enlace.href = url;
    enlace.download = "conocimiento-senas-knn.json";
    enlace.click();
    URL.revokeObjectURL(url);
}

function importarModelo(event) {
    const archivo = event.target.files[0];
    if (!archivo) return;

    const lector = new FileReader();
    lector.onload = (e) => {
        try {
            const datos = JSON.parse(e.target.result);
            restaurarDatasetDesdeObjeto(datos);
            guardarConocimiento();
            actualizarProgreso();
            document.getElementById("estadoEntrenamiento").textContent =
                "Conocimiento importado correctamente desde el archivo JSON.";
        } catch (error) {
            document.getElementById("estadoEntrenamiento").textContent =
                "Error al importar el conocimiento: " + error.message;
        }
    };
    lector.readAsText(archivo);
}

function reiniciarConocimiento() {
    classifier.clearAllClasses();
    Object.keys(ejemplosPorClase).forEach(k => delete ejemplosPorClase[k]);
    localStorage.removeItem(STORAGE_KEY);
    actualizarProgreso();
    document.getElementById("estadoEntrenamiento").textContent =
        "Entrenamiento eliminado. Puedes capturar nuevos ejemplos.";
}

function hablarPalabra() {
    const texto = palabraFormada.trim();
    if (!texto) {
        document.getElementById("mensajePalabras").textContent =
            "Primero forma una palabra para poder escucharla.";
        return;
    }
    if ("speechSynthesis" in window) {
        const voz = new SpeechSynthesisUtterance(texto.toLowerCase());
        voz.lang = "es-MX";
        window.speechSynthesis.speak(voz);
    }
}

function cargarPalabrasGuardadas() {
    try {
        const contenido = localStorage.getItem(WORDS_STORAGE_KEY);
        if (!contenido) {
            renderizarPalabrasGuardadas();
            return;
        }

        const guardadas = JSON.parse(contenido);
        if (!Array.isArray(guardadas) || !guardadas.every(
            palabra => typeof palabra === "string"
        )) {
            throw new Error("La lista guardada no tiene un formato válido.");
        }

        palabrasGuardadas = guardadas;
        renderizarPalabrasGuardadas();
    } catch (error) {
        console.error("No se pudieron recuperar las palabras:", error);
        document.getElementById("mensajePalabras").textContent =
            "No se pudieron recuperar las palabras guardadas: " + error.message;
    }
}

function guardarPalabra() {
    const palabra = palabraFormada.trim();
    if (!palabra) {
        document.getElementById("mensajePalabras").textContent =
            "Primero forma una palabra con las señas.";
        return;
    }

    if (palabrasGuardadas.includes(palabra)) {
        document.getElementById("mensajePalabras").textContent =
            "Esa palabra ya está en tu vocabulario.";
        return;
    }

    palabrasGuardadas.push(palabra);
    persistirPalabrasGuardadas();
}

function borrarPalabraGuardada(indice) {
    palabrasGuardadas.splice(indice, 1);
    persistirPalabrasGuardadas();
}

function limpiarPalabrasGuardadas() {
    palabrasGuardadas = [];
    persistirPalabrasGuardadas();
}

function persistirPalabrasGuardadas() {
    try {
        localStorage.setItem(WORDS_STORAGE_KEY, JSON.stringify(palabrasGuardadas));
        renderizarPalabrasGuardadas();
        document.getElementById("mensajePalabras").textContent =
            palabrasGuardadas.length
                ? "Tu vocabulario se guardó en este navegador."
                : "Aquí aparecerán las palabras que guardes.";
    } catch (error) {
        console.error("No se pudieron guardar las palabras:", error);
        document.getElementById("mensajePalabras").textContent =
            "No se pudieron guardar las palabras en este navegador: " +
            error.message;
    }
}

function renderizarPalabrasGuardadas() {
    const lista = document.getElementById("listaPalabras");
    lista.replaceChildren();

    palabrasGuardadas.forEach((palabra, indice) => {
        const elemento = document.createElement("li");
        elemento.className = "saved-word";

        const texto = document.createElement("span");
        texto.textContent = palabra + " ";

        const boton = document.createElement("button");
        boton.type = "button";
        boton.textContent = "×";
        boton.setAttribute("aria-label", "Eliminar " + palabra);
        boton.addEventListener("click", () => borrarPalabraGuardada(indice));

        elemento.append(texto, boton);
        lista.appendChild(elemento);
    });

    document.getElementById("limpiarPalabras").hidden =
        palabrasGuardadas.length === 0;
}

// Comprobar estabilidad de la predicción
function reconocerLetra(letra, confianza) {

    if (confianza < CONFIANZA_MINIMA) {
        letraCandidata = null;
        contadorEstabilidad = 0;

        document.getElementById("letraActual").textContent =
            "Seña no identificada con suficiente confianza.";

        return;
    }

    if (letra === "SIN_SEÑA") {
        letraCandidata = null;
        contadorEstabilidad = 0;
        ultimaLetraAgregada = null;

        document.getElementById("letraActual").textContent =
            "Esperando una seña...";

        return;
    }

    if (letra === letraCandidata) {
        contadorEstabilidad++;
    } else {
        letraCandidata = letra;
        contadorEstabilidad = 1;
    }

    document.getElementById("letraActual").textContent =
        "Detectando: " + letra +
        " (" + contadorEstabilidad + "/" +
        FRAMES_NECESARIOS + ")";

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
    document.getElementById("palabra").classList.remove("is-empty");

    document.getElementById("letraActual").textContent =
        "Letra reconocida: " + letra;
}

function agregarEspacio() {
    if (palabraFormada && !palabraFormada.endsWith(" ")) {
        palabraFormada += " ";
        document.getElementById("palabra").textContent = palabraFormada;
        document.getElementById("palabra").classList.remove("is-empty");
    }
}

function borrarUltimaLetra() {
    if (palabraFormada.length > 0) {
        palabraFormada = palabraFormada.slice(0, -1);
        const el = document.getElementById("palabra");
        if (palabraFormada.length === 0) {
            el.textContent = "Empieza a deletrear…";
            el.classList.add("is-empty");
        } else {
            el.textContent = palabraFormada;
        }
        ultimaLetraAgregada = null;
    }
}

// Borrar palabra
function borrarPalabra() {
    palabraFormada = "";

    document.getElementById("palabra").textContent = "Empieza a deletrear…";
    document.getElementById("palabra").classList.add("is-empty");
    document.getElementById("letraActual").textContent =
        "Esperando una seña...";

    ultimaLetraAgregada = null;
    letraCandidata = null;
    contadorEstabilidad = 0;
}

// Ejecutar aplicación
app();