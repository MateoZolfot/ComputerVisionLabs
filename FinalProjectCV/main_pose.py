import cv2
import socket
from ultralytics import YOLO

UDP_IP = "127.0.0.1" 
UDP_PORT = 5052
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Cargando modelo YOLOv8-Pose...")
model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0)

UMBRAL_X = 0.15  
UMBRAL_SALTO = 0.15 

print(f"🚀 Servidor Multijugador activado en el puerto {UDP_PORT}...")

while True:
    success, img = cap.read()
    if not success: break
    
    img = cv2.flip(img, 1) # Espejo
    results = model(img, verbose=False)
    
    jugadores_detectados = []
    
    for r in results:
        if r.keypoints is not None and len(r.keypoints.xyn) > 0:
            # Obtener todas las personas detectadas en este frame
            personas = r.keypoints.xyn.cpu().numpy()
            
            for kpts in personas:
                if len(kpts) >= 11:
                    mi_hombro = kpts[5] 
                    mi_muneca = kpts[9] 
                    
                    if mi_hombro[0] > 0 and mi_muneca[0] > 0:
                        # Guardar la persona con su posición X para ordenarlos luego
                        jugadores_detectados.append({
                            'x': mi_hombro[0],
                            'hombro': mi_hombro,
                            'muneca': mi_muneca
                        })
            
            break # Solo procesamos el primer resultado que contiene a todos
            
    # Ordenar de izquierda a derecha (por la coordenada X en pantalla)
    jugadores_detectados = sorted(jugadores_detectados, key=lambda p: p['x'])
    
    datos_a_enviar = []
    
    # Procesar máximo 2 jugadores
    for jugador in jugadores_detectados[:2]:
        hombro_x, hombro_y = jugador['hombro'][0], jugador['hombro'][1]
        muneca_x, muneca_y = jugador['muneca'][0], jugador['muneca'][1]
        
        target_move_x = 0
        jump = 0
        
        if muneca_x < hombro_x - UMBRAL_X: target_move_x = -1 
        elif muneca_x > hombro_x + UMBRAL_X: target_move_x = 1  
        
        if muneca_y < hombro_y - UMBRAL_SALTO: jump = 1
            
        datos_a_enviar.append(f"{target_move_x},{jump}")
    
    # Si hay jugadores, enviamos (ej: "-1,0|1,1" o solo "1,0")
    if datos_a_enviar:
        mensaje = "|".join(datos_a_enviar)
        sock.sendto(mensaje.encode(), (UDP_IP, UDP_PORT))
    
    img_annotated = results[0].plot()
    cv2.imshow("Motor Gestual - Multiplayer", img_annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()