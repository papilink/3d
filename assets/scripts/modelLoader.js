export async function loadModelParts() {
    try {
        // Cargar las dos partes del modelo
        const response1 = await fetch('assets/models/model_opt.tflite.partaa');
        const response2 = await fetch('assets/models/model_opt.tflite.partab');
        
        if (!response1.ok || !response2.ok) {
            throw new Error('No se pudieron cargar las partes del modelo');
        }

        // Convertir las respuestas a ArrayBuffer
        const buffer1 = await response1.arrayBuffer();
        const buffer2 = await response2.arrayBuffer();

        // Combinar los buffers
        const combinedBuffer = new Uint8Array(buffer1.byteLength + buffer2.byteLength);
        combinedBuffer.set(new Uint8Array(buffer1), 0);
        combinedBuffer.set(new Uint8Array(buffer2), buffer1.byteLength);

        // Crear un Blob con el modelo combinado
        const modelBlob = new Blob([combinedBuffer], { type: 'application/octet-stream' });
        const modelUrl = URL.createObjectURL(modelBlob);

        return modelUrl;
    } catch (error) {
        console.error('Error al cargar las partes del modelo:', error);
        throw error;
    }
}
