import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { loadModelParts } from './assets/scripts/modelLoader.js';

// --- Constants and Globals ---
const MODELS = {
    default: await loadModelParts(), // Cargar el modelo desde las partes
    fallback: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite'
};
let MODEL_URL = MODELS.default;
let tfliteModel = null;
let scene, camera, renderer, controls;

// Función para verificar si un archivo existe
async function checkFileExists(url) {
    try {
        const response = await fetch(url, { method: 'HEAD' });
        return response.ok;
    } catch (e) {
        console.warn('Error checking file:', e);
        return false;
    }
}
let currentMesh = null;

// --- DOM Elements ---
const uploadInput = document.getElementById('image-upload');
const generateBtn = document.getElementById('generate-btn');
const downloadBtn = document.getElementById('download-btn');
const loadingIndicator = document.getElementById('loading-indicator');
const canvas = document.getElementById('renderer-canvas');
const imagePreview = document.createElement('img');
imagePreview.style.display = 'none';
document.body.appendChild(imagePreview);

// --- Three.js Setup ---
function initThree() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x132F4C);
    const aspectRatio = canvas.clientWidth / canvas.clientHeight || 1;
    camera = new THREE.PerspectiveCamera(75, aspectRatio, 0.1, 1000);
    camera.position.z = 500;
    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setSize(canvas.parentElement.clientWidth, 500);
    renderer.setPixelRatio(window.devicePixelRatio);
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 10, 7.5);
    scene.add(directionalLight);
    window.addEventListener('resize', onWindowResize, false);
    animate();
}

function onWindowResize() {
    camera.aspect = canvas.parentElement.clientWidth / 500;
    camera.updateProjectionMatrix();
    renderer.setSize(canvas.parentElement.clientWidth, 500);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// --- AI & Core Logic ---
async function loadModel() {
    console.log('Iniciando verificación de modelos disponibles...');
    loadingIndicator.innerText = 'Inicializando sistema de IA...';
    loadingIndicator.style.display = 'block';
    
    try {
        // Intentar cargar el modelo desde las partes
        try {
            MODEL_URL = await loadModelParts();
        } catch (loadError) {
            console.error('Error al cargar partes del modelo:', loadError);
            console.log('Cambiando a modelo alternativo...');
            MODEL_URL = MODELS.fallback;
        }
        
        // Intentar cargar el modelo seleccionado
        console.log('Iniciando carga del modelo desde:', MODEL_URL);
        
        // Intentar la carga con timeout y reintentos
        const maxRetries = 3;
        let lastError = null;
        
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                const modelLoadPromise = tflite.loadTFLiteModel(MODEL_URL);
                const timeoutPromise = new Promise((_, reject) => 
                    setTimeout(() => reject(new Error(`Timeout en intento ${attempt}`)), 30000));
                
                tfliteModel = await Promise.race([modelLoadPromise, timeoutPromise]);
                
                if (tfliteModel) {
                    console.log('Modelo cargado exitosamente:', {
                        modelUrl: MODEL_URL,
                        intento: attempt,
                        tipo: MODEL_URL === MODELS.default ? 'Principal' : 'Alternativo'
                    });
                    loadingIndicator.innerText = 'Modelo listo. Puede comenzar a generar.';
                    return true;
                }
            } catch (error) {
                lastError = error;
                console.warn(`Intento ${attempt} fallido:`, error);
                
                if (attempt < maxRetries) {
                    loadingIndicator.innerText = `Reintentando cargar el modelo (${attempt}/${maxRetries})...`;
                    await new Promise(resolve => setTimeout(resolve, 1000 * attempt)); // Espera exponencial
                }
            }
        }
        
        // Si llegamos aquí, todos los intentos fallaron
        throw lastError || new Error('No se pudo cargar el modelo después de varios intentos');
        
    } catch (e) {
        console.error('Error detallado al cargar el modelo:', {
            message: e.message,
            stack: e.stack,
            modelUrl: MODEL_URL,
            browserInfo: navigator.userAgent
        });
        
        // Mostrar mensaje de error específico según el tipo de error
        let errorMessage = 'Error al cargar el modelo. ';
        if (e.message.includes('Timeout')) {
            errorMessage += 'La conexión es muy lenta. Intente con una mejor conexión a internet.';
        } else if (e.message.includes('CORS')) {
            errorMessage += 'Error de acceso al servidor. Intente más tarde.';
        } else if (e.message.includes('memory')) {
            errorMessage += 'No hay suficiente memoria disponible. Cierre otras aplicaciones e intente de nuevo.';
        } else {
            errorMessage += 'Por favor, verifique su conexión a internet y recargue la página.';
        }
        
        alert(errorMessage);
        loadingIndicator.innerText = 'Error al cargar el modelo. Intente recargar la página.';
        return false;
    }
}

async function estimateDepth(imgElement) {
    if (!tfliteModel) {
        console.error('Intento de usar el modelo antes de que esté cargado');
        alert('El modelo aún no está cargado. Por favor, espere a que se complete la carga.');
        return null;
    }
    console.log('Iniciando estimación de profundidad...', {
        modelStatus: tfliteModel ? 'Cargado' : 'No cargado',
        imageSize: `${imgElement.width}x${imgElement.height}`
    });
    loadingIndicator.innerText = 'Calculando profundidad de la imagen...';
    loadingIndicator.style.display = 'block';

    const tensor = tf.tidy(() => {
        let input = tf.browser.fromPixels(imgElement);
        const resized = tf.image.resizeBilinear(input, [256, 256]);
        const normalized = resized.div(255.0);
        const batched = normalized.expandDims(0);
        let output = tfliteModel.predict(batched);
        output = tf.squeeze(output);
        output = tf.div(tf.sub(output, tf.min(output)), tf.sub(tf.max(output), tf.min(output)));
        return output;
    });

    console.log('Depth estimation complete.');
    loadingIndicator.style.display = 'none';
    return tensor;
}

async function createMeshFromDepthMap(depthMapTensor, textureImage) {
    const depthMap = await depthMapTensor.array();
    const [height, width] = depthMapTensor.shape;
    
    // Obtener valores de configuración
    const depthScale = document.getElementById('depth-scale').value / 100;
    const baseHeight = document.getElementById('base-height').value / 100;
    const useNormalMap = document.getElementById('use-normal-map').checked;
    const autoCenter = document.getElementById('auto-center').checked;
    
    // Calcular escala de extrusión
    const extrusionScale = 100.0 * depthScale;
    
    // Crear geometría con base opcional
    const geometry = new THREE.PlaneGeometry(width, height, width - 1, height - 1);
    const positionAttribute = geometry.getAttribute('position');
    
    // Aplicar mapa de profundidad y base
    for (let i = 0; i < positionAttribute.count; i++) {
        const y = Math.floor(i / width);
        const depth = depthMap[y][i % width];
        const z = (1.0 - depth) * extrusionScale + (baseHeight * 20); // Base height
        positionAttribute.setZ(i, z);
    }
    
    // Computar normales para mejor iluminación
    geometry.computeVertexNormals();
    
    // Crear textura y material
    const texture = new THREE.Texture(textureImage);
    texture.needsUpdate = true;
    
    const materialParams = {
        map: texture,
        side: THREE.DoubleSide
    };
    
    // Agregar normal map si está activado
    if (useNormalMap) {
        const normalStrength = 0.5;
        materialParams.normalMap = texture;
        materialParams.normalScale = new THREE.Vector2(normalStrength, normalStrength);
    }
    
    const material = new THREE.MeshStandardMaterial(materialParams);
    
    // Crear mesh
    const mesh = new THREE.Mesh(geometry, material);
    mesh.rotation.x = -Math.PI / 2;
    
    // Centrar automáticamente si está activado
    if (autoCenter) {
        geometry.center();
        mesh.position.set(0, 0, 0);
    }
    
    return mesh;
}

// --- Exporter ---
// Función para validar el modelo antes de exportar
function validateModelForExport(mesh) {
    if (!mesh.geometry) {
        throw new Error('El modelo no tiene geometría válida');
    }
    
    const geometry = mesh.geometry;
    if (!geometry.attributes.position || geometry.attributes.position.count === 0) {
        throw new Error('El modelo no tiene vértices válidos');
    }
    
    if (!mesh.material || !mesh.material.map) {
        throw new Error('El modelo no tiene textura válida');
    }
    
    return true;
}

// Función para optimizar el modelo para exportación
function optimizeModelForExport(mesh) {
    const optimizedMesh = mesh.clone();
    
    // Centrar el modelo en el origen
    optimizedMesh.geometry.center();
    
    // Asegurarse de que las normales estén actualizadas
    optimizedMesh.geometry.computeVertexNormals();
    
    // Optimizar la geometría
    optimizedMesh.geometry.computeBoundingBox();
    optimizedMesh.geometry.computeBoundingSphere();
    
    return optimizedMesh;
}

async function downloadGLB() {
    try {
        if (!currentMesh) {
            throw new Error("No hay modelo para descargar. Por favor, genera un modelo primero.");
        }

        // Validar el modelo
        validateModelForExport(currentMesh);

        // Mostrar indicador de progreso
        loadingIndicator.style.display = 'block';
        loadingIndicator.innerText = 'Preparando modelo para descarga...';
        downloadBtn.disabled = true;

        // Optimizar el modelo para exportación
        const meshForExport = optimizeModelForExport(currentMesh);
        
        // Obtener el nombre del archivo original
        const originalFileName = uploadInput.files[0]?.name || 'image';
        const baseFileName = originalFileName.split('.')[0];
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        
        // Crear un nuevo objeto de escena para la exportación
        const exportScene = new THREE.Scene();
        
        // Ajustar la posición y rotación para una mejor vista por defecto
        meshForExport.position.set(0, 0, 0);
        meshForExport.rotation.set(-Math.PI / 2, 0, 0); // Orientación estándar para GLB
        exportScene.add(meshForExport);
        
        // Agregar luces a la escena de exportación
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 7.5);
        exportScene.add(ambientLight, directionalLight);
        
        // Configurar opciones de exportación
        const options = {
            binary: true,
            includeCustomExtensions: true,
            embedImages: true,
            animations: [],
            onlyVisible: true,
            maxTextureSize: 4096,
            forceIndices: true,
            truncateDrawRange: true,
            userData: {
                generatedBy: 'Papiweb 3D Converter',
                originalImage: originalFileName,
                createdAt: timestamp,
                version: '1.0',
                metadata: {
                    vertices: meshForExport.geometry.attributes.position.count,
                    faces: meshForExport.geometry.index ? meshForExport.geometry.index.count / 3 : 0,
                    textureResolution: meshForExport.material.map.image ? 
                        `${meshForExport.material.map.image.width}x${meshForExport.material.map.image.height}` : 'N/A'
                }
            }
        };

        // Actualizar progreso
        loadingIndicator.innerText = 'Exportando modelo...';

        // Exportar usando Promise
        const result = await new Promise((resolve, reject) => {
            const exporter = new GLTFExporter();
            exporter.parse(
                exportScene,
                (gltf) => resolve(gltf),
                (error) => reject(error),
                options
            );
        });

        // Crear y preparar el archivo para descarga
        const blob = new Blob([result], { type: 'model/gltf-binary' });
        const url = URL.createObjectURL(blob);
        
        // Crear elemento de descarga con estilo
        const downloadWrapper = document.createElement('div');
        downloadWrapper.style.position = 'fixed';
        downloadWrapper.style.bottom = '20px';
        downloadWrapper.style.right = '20px';
        downloadWrapper.style.padding = '15px';
        downloadWrapper.style.background = 'rgba(33, 150, 243, 0.9)';
        downloadWrapper.style.borderRadius = '8px';
        downloadWrapper.style.boxShadow = '0 2px 8px rgba(0,0,0,0.2)';
        downloadWrapper.style.color = 'white';
        downloadWrapper.style.zIndex = '1000';
        downloadWrapper.innerHTML = '⬇️ Descarga iniciada...';
        
        document.body.appendChild(downloadWrapper);

        // Iniciar descarga
        const link = document.createElement('a');
        link.href = url;
        link.download = `${baseFileName}_3d_${timestamp}.glb`;
        link.click();
        
        // Mostrar mensaje de éxito
        loadingIndicator.innerText = 'Modelo exportado con éxito';
        
        // Limpiar recursos
        setTimeout(() => {
            URL.revokeObjectURL(url);
            exportScene.remove(meshForExport, ambientLight, directionalLight);
            meshForExport.geometry.dispose();
            meshForExport.material.dispose();
            document.body.removeChild(downloadWrapper);
        }, 3000);

    } catch (error) {
        console.error('Error durante la exportación GLB:', error);
        alert(error.message || 'Error al exportar el modelo. Por favor, intenta de nuevo.');
    } finally {
        // Restaurar interfaz
        loadingIndicator.style.display = 'none';
        downloadBtn.disabled = false;
    }
}

// Función para actualizar la geometría del mesh
async function updateMeshGeometry() {
    if (!currentMesh || !imagePreview.complete) return;
    
    try {
        const depthMapTensor = await estimateDepth(imagePreview);
        if (depthMapTensor) {
            scene.remove(currentMesh);
            currentMesh.geometry.dispose();
            currentMesh.material.dispose();
            
            currentMesh = await createMeshFromDepthMap(depthMapTensor, imagePreview);
            const boundingBox = new THREE.Box3().setFromObject(currentMesh);
            const center = boundingBox.getCenter(new THREE.Vector3());
            const size = boundingBox.getSize(new THREE.Vector3());
            
            controls.target.copy(center);
            camera.position.z = Math.max(size.x, size.y, size.z) * 1.5;
            camera.lookAt(center);
            
            scene.add(currentMesh);
            depthMapTensor.dispose();
        }
    } catch (error) {
        console.error('Error al actualizar la geometría:', error);
    }
}

// --- Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
    // Configurar controles de ajuste
    const depthScale = document.getElementById('depth-scale');
    const depthValue = document.getElementById('depth-value');
    const baseHeight = document.getElementById('base-height');
    const baseValue = document.getElementById('base-value');
    
    // Actualizar valores mostrados
    depthScale.addEventListener('input', () => {
        depthValue.textContent = `${depthScale.value}%`;
        if (currentMesh) {
            updateMeshGeometry();
        }
    });
    
    baseHeight.addEventListener('input', () => {
        baseValue.textContent = `${baseHeight.value}%`;
        if (currentMesh) {
            updateMeshGeometry();
        }
    });
    
    document.getElementById('auto-center').addEventListener('change', () => {
        if (currentMesh) {
            updateMeshGeometry();
        }
    });
    
    document.getElementById('use-normal-map').addEventListener('change', () => {
        if (currentMesh) {
            updateMeshGeometry();
        }
    });

    // Verificar requisitos del navegador
    if (!window.WebGLRenderingContext) {
        alert('Tu navegador no soporta WebGL, necesario para la visualización 3D.');
        return;
    }

    if (typeof tf === 'undefined') {
        console.error('TensorFlow.js no se ha cargado correctamente');
        alert('Error: No se pudo cargar la biblioteca de IA. Por favor, recarga la página.');
        return;
    }

    // Inicializar Three.js
    initThree();

    // Verificar y cargar TFLite
    let attempts = 0;
    const maxAttempts = 50; // 5 segundos máximo

    function waitForTFLite() {
        if (typeof tflite !== 'undefined') {
            console.log('TFLite detectado, iniciando carga del modelo...');
            loadModel().then(success => {
                if (!success) {
                    console.error('No se pudo cargar ningún modelo');
                }
            });
        } else {
            attempts++;
            if (attempts >= maxAttempts) {
                console.error('TFLite no se pudo cargar después de varios intentos');
                alert('Error: No se pudo inicializar el sistema de IA. Por favor, recargue la página o intente con otro navegador.');
                return;
            }
            console.log(`Esperando que TFLite esté disponible... (intento ${attempts}/${maxAttempts})`);
            setTimeout(waitForTFLite, 100);
        }
    }
    waitForTFLite();

    generateBtn.addEventListener('click', () => {
        const file = uploadInput.files[0];
        if (!file) {
            alert('Please select an image file first.');
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.onload = async () => {
                const depthMapTensor = await estimateDepth(imagePreview);
                if (depthMapTensor) {
                    if (currentMesh) {
                        scene.remove(currentMesh);
                        currentMesh.geometry.dispose();
                        currentMesh.material.dispose();
                    }
                    currentMesh = await createMeshFromDepthMap(depthMapTensor, imagePreview);
                    const boundingBox = new THREE.Box3().setFromObject(currentMesh);
                    const center = boundingBox.getCenter(new THREE.Vector3());
                    const size = boundingBox.getSize(new THREE.Vector3());
                    controls.target.copy(center);
                    camera.position.z = Math.max(size.x, size.y, size.z) * 1.5;
                    camera.lookAt(center);
                    scene.add(currentMesh);
                    canvas.style.display = 'block';
                    downloadBtn.style.display = 'inline-block';
                    depthMapTensor.dispose();
                }
            };
        };
        reader.readAsDataURL(file);
    });

    downloadBtn.addEventListener('click', downloadGLB);
});
