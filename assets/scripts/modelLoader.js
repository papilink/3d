// Cache key para almacenar el modelo en IndexedDB
const MODEL_CACHE_KEY = 'papiweb3d_model_cache';
const MODEL_VERSION = '1.0';

// Función para verificar si el modelo está en caché
async function checkModelCache() {
    try {
        if ('caches' in window) {
            const cache = await caches.open('papiweb3d-model-cache');
            const parts = await Promise.all([
                cache.match('model_part1'),
                cache.match('model_part2')
            ]);
            return parts[0] && parts[1] ? parts : null;
        }
    } catch (error) {
        console.warn('Error checking cache:', error);
    }
    return null;
}

// Función para guardar el modelo en caché
async function cacheModelParts(part1Buffer, part2Buffer) {
    try {
        if ('caches' in window) {
            const cache = await caches.open('papiweb3d-model-cache');
            await Promise.all([
                cache.put('model_part1', new Response(part1Buffer)),
                cache.put('model_part2', new Response(part2Buffer))
            ]);
        }
    } catch (error) {
        console.warn('Error caching model:', error);
    }
}

// Función para mostrar el progreso de carga
function updateLoadingProgress(progress) {
    const loadingIndicator = document.getElementById('loading-indicator');
    if (loadingIndicator) {
        loadingIndicator.innerText = `Cargando modelo: ${Math.round(progress)}%`;
        loadingIndicator.style.display = 'block';
    }
}

// Función para cargar una parte del modelo con progreso
async function loadModelPartWithProgress(url, partNumber, totalSize) {
    const response = await fetch(url);
    const reader = response.body.getReader();
    const contentLength = parseInt(response.headers.get('Content-Length') || '0');
    
    let receivedLength = 0;
    const chunks = [];
    
    while(true) {
        const {done, value} = await reader.read();
        
        if (done) {
            break;
        }
        
        chunks.push(value);
        receivedLength += value.length;
        
        // Calcular y mostrar el progreso
        const partProgress = (receivedLength / contentLength) * 100;
        const totalProgress = ((partNumber - 1) * 50) + (partProgress / 2);
        updateLoadingProgress(totalProgress);
    }
    
    const chunksAll = new Uint8Array(receivedLength);
    let position = 0;
    for(let chunk of chunks) {
        chunksAll.set(chunk, position);
        position += chunk.length;
    }
    
    return chunksAll.buffer;
}

// Función principal para cargar el modelo
export async function loadModelParts() {
    try {
        // Verificar caché primero
        const cachedParts = await checkModelCache();
        if (cachedParts) {
            console.log('Usando modelo en caché');
            const [part1Response, part2Response] = cachedParts;
            const [buffer1, buffer2] = await Promise.all([
                part1Response.arrayBuffer(),
                part2Response.arrayBuffer()
            ]);
            
            const combinedBuffer = new Uint8Array(buffer1.byteLength + buffer2.byteLength);
            combinedBuffer.set(new Uint8Array(buffer1), 0);
            combinedBuffer.set(new Uint8Array(buffer2), buffer1.byteLength);
            
            return URL.createObjectURL(new Blob([combinedBuffer], { type: 'application/octet-stream' }));
        }

        // Si no está en caché, cargar las partes
        console.log('Cargando modelo desde el servidor...');
        updateLoadingProgress(0);

        // Cargar las dos partes con progreso
        const [buffer1, buffer2] = await Promise.all([
            loadModelPartWithProgress('assets/models/model_opt.tflite.partaa', 1),
            loadModelPartWithProgress('assets/models/model_opt.tflite.partab', 2)
        ]);

        // Verificar integridad de las partes
        if (!buffer1 || !buffer2) {
            throw new Error('Error de integridad en las partes del modelo');
        }

        // Combinar los buffers
        const combinedBuffer = new Uint8Array(buffer1.byteLength + buffer2.byteLength);
        combinedBuffer.set(new Uint8Array(buffer1), 0);
        combinedBuffer.set(new Uint8Array(buffer2), buffer1.byteLength);

        // Guardar en caché para uso futuro
        await cacheModelParts(buffer1, buffer2);

        // Crear y retornar URL del modelo combinado
        const modelBlob = new Blob([combinedBuffer], { type: 'application/octet-stream' });
        const modelUrl = URL.createObjectURL(modelBlob);

        // Ocultar indicador de carga
        updateLoadingProgress(100);
        setTimeout(() => {
            const loadingIndicator = document.getElementById('loading-indicator');
            if (loadingIndicator) {
                loadingIndicator.style.display = 'none';
            }
        }, 1000);

        return modelUrl;
    } catch (error) {
        console.error('Error al cargar las partes del modelo:', error);
        const loadingIndicator = document.getElementById('loading-indicator');
        if (loadingIndicator) {
            loadingIndicator.innerText = 'Error al cargar el modelo. Intente recargar la página.';
        }
        throw error;
    }
}
