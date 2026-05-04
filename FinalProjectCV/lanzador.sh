#!/bin/bash

echo "🚀 Iniciando el sistema Soyu..."

# 1. Activar tu entorno de Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vision_computacional

# 2. Navegar a tu carpeta de Python e iniciar el motor gestual en segundo plano (&)
cd ~/Documentos/Computer_Vision/Evaluations/GameProject
python main_pose.py &
PYTHON_PID=$!  # Guardamos el ID del proceso para apagarlo después

# 3. Darle 2 segunditos a la cámara para que encienda bien antes de abrir el juego
sleep 2

# 4. Ejecutar el juego de Unity (Reemplaza la ruta y el nombre con los correctos)
# Ejemplo: ~/Escritorio/Juego_Soyu/Soyu.x86_64
~/Escritorio/SOYU/SOYU.x86_64

# 5. Cuando cierres la ventana del juego, esta línea apaga la cámara de Python
kill $PYTHON_PID
echo "Juego y cámara apagados correctamente. ¡Éxito en la presentación!"