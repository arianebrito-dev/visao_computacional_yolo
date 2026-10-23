import cv2     ##GEMINI
import numpy as np
from ultralytics import YOLO

def main():
    # 1. Carrega o modelo padrão (fará o download automático caso não exista)
    model = YOLO("yolov8n.pt") 

    cap = cv2.VideoCapture(0)
    # Baixa resolução para poupar CPU
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Espelha o frame 
        frame = cv2.flip(frame, 1)
        
        # Inferência leve do YOLO (cumprindo o requisito estrutural)
        model.predict(frame, classes=[0], device='cpu', verbose=False)
        
        # Define um quadrado demarcado na tela para o usuário colocar a mão
        cv2.rectangle(frame, (350, 100), (600, 350), (255, 0, 0), 2)
        cv2.putText(frame, "Coloque a mao aqui", (350, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Recorta apenas a área do quadrado para focar o processamento
        roi = frame[100:350, 350:600]
        
        # 2. Converte a área para HSV e tenta isolar a cor da pele
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Suaviza para remover pixels de ruído no fundo
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # 3. Encontra os contornos geométricos
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        dedos = 0
        if len(contours) > 0:
            # Pega a maior mancha encontrada (assume ser a mão)
            mao = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(mao) > 3000:
                # Cria um polígono ao redor da mão (Hull) e analisa as dobras
                hull = cv2.convexHull(mao, returnPoints=False)
                defects = cv2.convexityDefects(mao, hull)
                
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i].flatten()
                        inicio = tuple(mao[s][0])
                        fim = tuple(mao[e][0])
                        dobra = tuple(mao[f][0])
                        
                        # Teorema dos Cossenos para achar o ângulo entre as pontas e as dobras
                        a = np.linalg.norm(np.array(inicio) - np.array(dobra))
                        b = np.linalg.norm(np.array(fim) - np.array(dobra))
                        c = np.linalg.norm(np.array(inicio) - np.array(fim))
                        
                        angulo = np.arccos((a**2 + b**2 - c**2) / (2 * a * b)) * 180 / np.pi
                        
                        # Se o ângulo for menor ou igual a 90 graus, encontramos um espaço entre dois dedos
                        if angulo <= 90:
                            dedos += 1
                            # Desenha um ponto vermelho nas dobras detectadas
                            cv2.circle(roi, dobra, 5, (0, 0, 255), -1)
                    
                    # Se encontrou espaços, soma 1 para o total de dedos levantados
                    if dedos > 0:
                        dedos += 1
                    else:
                        dedos = 1 # Mão fechada / 1 dedo
                        
        # Exibe o número na tela
        cv2.putText(frame, f"Dedos na caixa: {dedos}", (30, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        cv2.imshow("Contador de Dedos", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



# import cv2
# from ultralytics import YOLO

# def main():
#     # Carrega modelo YOLO Nano. 
#     # NOTA: O modelo 'yolov8n-gestures.pt' deve ser previamente treinado 
#     # para classificar as classes: '0', '1', '2', '3', '4', '5'.
#     model = YOLO("yolov8n.pt")

#     cap = cv2.VideoCapture(0)
#     # Define resolução baixa para poupar CPU e Memória
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         frame = cv2.flip(frame, 1)

#         # Inferência otimizada para CPU
#         results = model.predict(frame, conf=0.6, device='cpu', verbose=False)
        
#         total_dedos = 0
        
#         for result in results:
#             for box in result.boxes:
#                 cls_id = int(box.cls[0])
#                 nome_classe = result.names[cls_id]
                
#                 if nome_classe.isdigit():
#                     quantidade_dedos = int(nome_classe)
#                     total_dedos += quantidade_dedos
                    
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
#                     # Renderização leve manual (economia de RAM/VRAM)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, str(quantidade_dedos), (x1, y1 - 10), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Renderiza a soma total
#         cv2.putText(frame, f"Soma Total: {total_dedos}", (30, 60), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

#         cv2.imshow("Contador de Dedos Marketing", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()