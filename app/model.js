// ==========================================
//   *** CONFIGURACIÓN GLOBAL AJUSTABLE ***
// ==========================================

// --- CONFIGURACIÓN MODELO & DETECCIÓN ---
const MODEL_PATH = "models/yolo.onnx";
const INPUT_SIZE = 640;
const SCORE_THRESHOLD = 0.01; 
const NMS_THRESHOLD = 0.45;   

// --- CONFIGURACIÓN DE RASTREO Y VELOCIDAD ---
const FPS_ASSUMED = 8;        // Frames por segundo del video
const PX_PER_METER = 40;      // 
const MAX_TRACKING_DISTANCE = 400; // Máxima distancia en píxeles que puede saltar un objeto entre frames
const STABILITY_THRESHOLD = 3; // N° de frames seguidos que debe ser visto para ser considerado estable 

const SPEED_LIMIT_KMH = 50

// ==========================================
//   VARIABLES GLOBALES
// ==========================================
let session;
let isPlaying = false;
const violations = []; 

// ==========================================
//   CLASE TRACKER (RASTREO DE PLACAS)
// ==========================================
class VehicleTracker {
    constructor() {
        this.vehicles = {};
        this.nextId = 1;
        this.maxDistance = MAX_TRACKING_DISTANCE; 
    }

    update(detections) {
        const currentFrameVehicles = [];

        detections.forEach(det => {
            const center = { x: (det.x1 + det.x2) / 2, y: (det.y1 + det.y2) / 2 };

            let bestMatchId = null;
            let minDist = Infinity;

            for (const [id, v] of Object.entries(this.vehicles)) {
                if (v.lost > 5) continue;

                const dist = Math.hypot(center.x - v.center.x, center.y - v.center.y);
                if (dist < minDist && dist < this.maxDistance) {
                    minDist = dist;
                    bestMatchId = id;
                }
            }

            if (bestMatchId) {
                const v = this.vehicles[bestMatchId];
                const distanceMeters = Math.abs(center.y - v.center.y) / PX_PER_METER;
                const speedKmh = distanceMeters * FPS_ASSUMED * 3.6;

                v.speed = (v.speed * 0.6) + (speedKmh * 0.4);
                
                v.center = center;
                v.box = det;
                v.lost = 0;
                v.framesTracked++;
                currentFrameVehicles.push(v);
            } else {
                const newId = this.nextId++;
                this.vehicles[newId] = {
                    id: newId, center: center, box: det, speed: 0, framesTracked: 0, lost: 0, processed: false
                };
                currentFrameVehicles.push(this.vehicles[newId]);
            }
        });

        for (const id in this.vehicles) {
            const v = this.vehicles[id];
            if (!currentFrameVehicles.includes(v)) {
                v.lost++;
                if (v.lost > 20) delete this.vehicles[id];
            }
        }
        return this.vehicles;
    }
}

const tracker = new VehicleTracker();

// ==========================================
//   CARGA DEL MODELO (ONNX)
// ==========================================
async function loadModel() {
    try {
        session = await ort.InferenceSession.create(MODEL_PATH, { executionProviders: ['wasm'] });
        console.log("✅ MODELO: Modelo cargado exitosamente.");
        document.getElementById("status").innerText = "Modelo listo. Carga un video.";
        document.getElementById("status").style.color = "green";
    } catch (e) {
        console.error("❌ ERROR: No se pudo cargar el modelo ONNX. Verifica la ruta.", e);
        document.getElementById("status").innerText = "Error al cargar modelo ❌";
    }
}
loadModel();

// ==========================================
//   CONTROL DE VIDEO E INTERFAZ
// ==========================================
const video = document.getElementById("videoInput");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

document.getElementById("fileInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
        const url = URL.createObjectURL(file);
        video.src = url;
    }
});

video.addEventListener("loadedmetadata", () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    console.log(`[🎥 VIDEO] Resolución detectada: ${canvas.width}x${canvas.height}. Canvas ajustado.`);
});

function startProcessing() {
    if (!session || !video.src) {
        alert("Carga un video primero y espera al modelo.");
        return;
    }
    
    if (video.videoWidth > 0) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }

    video.play();
    isPlaying = true;
    processFrame();
}

function stopProcessing() {
    video.pause();
    isPlaying = false;
}

// ==========================================
//   BUCLE PRINCIPAL (FRAME BY FRAME)
// ==========================================
async function processFrame() {
    if (!isPlaying || video.paused || video.ended) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const { tensor, scale, dx, dy } = preprocess(video);

    const input = { [session.inputNames[0]]: tensor };
    const output = await session.run(input);
    const outputTensor = output[session.outputNames[0]].data;
    const dims = output[session.outputNames[0]].dims;

    const detections = parseYOLOOutput(outputTensor, dims, scale, dx, dy);

    const trackedVehicles = tracker.update(detections);
    drawScene(trackedVehicles);

    requestAnimationFrame(processFrame);
}

// ==========================================
//   DIBUJO Y LÓGICA DE INFRACCIONES
// ==========================================
function drawScene(vehicles) {
    for (const id in vehicles) {
        const v = vehicles[id];
        if (v.lost > 0) continue;

        const { x1, y1, x2, y2 } = v.box;
        
        let color = "#00FF00";
        let text = `ID:${v.id} ${v.speed.toFixed(0)}km/h`;

        // 1. Condición de Estabilidad (Trigger de OCR)
        if (v.framesTracked >= STABILITY_THRESHOLD) { 
            
            if (v.speed > SPEED_LIMIT_KMH) {
                color = "#FF0000";
                text = `🚨 ${v.speed.toFixed(0)} km/h`;
            } else {
                color = "#00FF00"; // Si no supera el límite visual, pero es estable
            }

            // --- DETECTAR INFRACCIÓN (OCR TRIGGER) ---
            if (!v.processed) { 
                v.processed = true; 
                console.log(`🚨 INFRACCIÓN: ID ${v.id} Tasa estable. Enviando a OCR...`);
                processViolation(v);
            }
        } else {
            color = "#00FFFF"; // Cyan
            text = `Rastreo... (${v.framesTracked}/${STABILITY_THRESHOLD}f)`;
        }

        // Dibujar Caja
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Dibujar Texto
        ctx.fillStyle = color;
        ctx.font = "bold 24px Arial";
        ctx.fillText(text, x1, y1 - 10);
    }
}

// ==========================================
//   PROCESAR INFRACCIÓN (OCR)
// ==========================================
async function processViolation(vehicle) {
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    tempCanvas.getContext("2d").drawImage(canvas, 0, 0);

    const plateText = await extractPlateText(tempCanvas, vehicle.box);

    if (plateText !== "?" && plateText !== "Err") {
        const record = {
            id: vehicle.id, plate: plateText, speed: vehicle.speed.toFixed(2), timestamp: new Date().toLocaleTimeString()
        };
        
        violations.push(record);
        updateViolationsUI();
        console.log(`✅ OCR: Placa ${plateText} guardada.`);
    } else {
        console.warn(`⚠️ OCR FALLIDO para ID ${vehicle.id}. Resultado: ${plateText}.`);
    }
}

function updateViolationsUI() {
    const list = document.getElementById("violationsLog");
    if(list) {
        const last = violations[violations.length - 1];
        const item = document.createElement("li");
        item.innerHTML = `<b>${last.plate}</b> - <span style="color:red">${last.speed} km/h</span> - ${last.timestamp}`;
        list.prepend(item);
    }
}

// ==========================================
//   EXPORTAR TXT
// ==========================================
function downloadReport() {
    if (violations.length === 0) {
        alert("No hay infracciones registradas.");
        return;
    }

    let content = "REPORTE DE INFRACCIONES DE TRÁFICO\n";
    content += `Fecha: ${new Date().toLocaleDateString()}\n`;
    content += "===================================\n\n";

    violations.forEach(v => {
        content += `HORA: ${v.timestamp}\n`;
        content += `ID: ${v.id}\n`;
        content += `PLACA: ${v.plate}\n`;
        content += `VELOCIDAD: ${v.speed} km/h\n`;
        content += "-----------------------------------\n";
    });

    const blob = new Blob([content], { type: "text/plain" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `multas_${Date.now()}.txt`;
    link.click();
}

// ==========================================
//   FUNCIONES DE SOPORTE YOLO (PREPROCESS)
// ==========================================
function preprocess(source) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    const INPUT_SIZE = 640; 
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;

    const w = source.videoWidth || source.width;
    const h = source.videoHeight || source.height;

    const scale = Math.min(INPUT_SIZE / w, INPUT_SIZE / h);
    const newW = Math.round(w * scale);
    const newH = Math.round(h * scale);
    
    const dx = (INPUT_SIZE - newW) / 2;
    const dy = (INPUT_SIZE - newH) / 2;

    ctx.fillStyle = "#808080";
    ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    ctx.drawImage(source, dx, dy, newW, newH);

    const imgData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const data = imgData.data;

    const float32Data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
        float32Data[j] = data[i] / 255.0;
        float32Data[j + INPUT_SIZE * INPUT_SIZE] = data[i + 1] / 255.0;
        float32Data[j + 2 * INPUT_SIZE * INPUT_SIZE] = data[i + 2] / 255.0;
    }

    const tensor = new ort.Tensor("float32", float32Data, [1, 3, INPUT_SIZE, INPUT_SIZE]);

    return { tensor, scale, dx, dy };
}

// ==========================================
//   PARSE YOLO OUTPUT (UNIVERSAL)
// ==========================================
function parseYOLOOutput(data, dims, scale, dx, dy) {
    const detections = [];
    const SCORE_THRESHOLD = 0.01; // Usa la variable global
    const NMS_THRESHOLD = 0.45;

    let numAnchors, numClassProps, isTransposed;

    if (dims[2] === 8400) {
        isTransposed = false;
        numAnchors = dims[2];
        numClassProps = dims[1];
    } else {
        isTransposed = true;
        numAnchors = dims[1];
        numClassProps = dims[2];
    }

    for (let i = 0; i < numAnchors; i++) {
        let score, xc, yc, w, h;

        if (!isTransposed) {
            xc = data[i];
            yc = data[i + numAnchors];
            w  = data[i + 2 * numAnchors];
            h  = data[i + 3 * numAnchors];
            score = data[i + 4 * numAnchors];
        } else {
            const rowOffset = i * numClassProps;
            xc = data[rowOffset + 0];
            yc = data[rowOffset + 1];
            w  = data[rowOffset + 2];
            h  = data[rowOffset + 3];
            score = data[rowOffset + 4];
        }

        if (score < SCORE_THRESHOLD) continue;

        let x1 = (xc - w / 2 - dx) / scale;
        let y1 = (yc - h / 2 - dy) / scale;
        let x2 = (xc + w / 2 - dx) / scale;
        let y2 = (yc + h / 2 - dy) / scale;

        detections.push({ x1, y1, x2, y2, score });
    }
    
    return nms(detections, NMS_THRESHOLD);
}

// ==========================================
//   ALGORITMO NMS
// ==========================================
function nms(detections, iouThreshold) {
    detections.sort((a, b) => b.score - a.score);
    const selected = [];
    const active = new Array(detections.length).fill(true);

    for (let i = 0; i < detections.length; i++) {
        if (!active[i]) continue;
        selected.push(detections[i]);

        for (let j = i + 1; j < detections.length; j++) {
            if (!active[j]) continue;

            const boxA = detections[i];
            const boxB = detections[j];

            const x1 = Math.max(boxA.x1, boxB.x1);
            const y1 = Math.max(boxA.y1, boxB.y1);
            const x2 = Math.min(boxA.x2, boxB.x2);
            const y2 = Math.min(boxA.y2, boxB.y2); 

            const interW = Math.max(0, x2 - x1);
            const interH = Math.max(0, y2 - y1);
            const interArea = interW * interH;

            const areaA = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1);
            const areaB = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1);

            const iou = interArea / (areaA + areaB - interArea);

            if (iou > iouThreshold) {
                active[j] = false;
            }
        }
    }
    return selected;
}

// ==========================================
//   OCR UNIVERSAL (GENÉRICO)
// ==========================================
async function extractPlateText(sourceImg, det) {
    const cropCanvas = document.createElement("canvas");
    const cropCtx = cropCanvas.getContext("2d");

    // Margen 10%
    const padX = (det.x2 - det.x1) * 0.1;
    const padY = (det.y2 - det.y1) * 0.1;

    const x = Math.max(0, det.x1 - padX);
    const y = Math.max(0, det.y1 - padY);
    const w = Math.min(sourceImg.width - x, (det.x2 - det.x1) + 2 * padX);
    const h = Math.min(sourceImg.height - y, (det.y2 - det.y1) + 2 * padY);

    cropCanvas.width = w * 2;
    cropCanvas.height = h * 2;
    
    cropCtx.imageSmoothingEnabled = true; 
    cropCtx.drawImage(sourceImg, x, y, w, h, 0, 0, cropCanvas.width, cropCanvas.height);

    makeGrayscale(cropCtx, cropCanvas.width, cropCanvas.height);

    try {
        const result = await Tesseract.recognize(
            cropCanvas,
            'eng', 
            {
                tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                tessedit_pageseg_mode: '6' 
            }
        );

        let text = result.data.text.trim().replace(/[^A-Z0-9]/g, '');

        if (text.length < 3 || text.length > 10) return "?";

        return text;

    } catch (error) {
        console.error("OCR Error:", error);
        return "Err";
    }
}

function makeGrayscale(ctx, width, height) {
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;
    for (let i = 0; i < data.length; i += 4) {
        const gray = 0.2126 * data[i] + 0.7152 * data[i+1] + 0.0722 * data[i+2];
        data[i] = data[i+1] = data[i+2] = gray;
    }
    ctx.putImageData(imgData, 0, 0);
}
