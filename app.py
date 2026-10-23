import cv2
from ultralytics import YOLO

def iniciar_deteccao_webcam():
    """
    Abre a webcam padrão do computador e realiza detecção de objetos/pessoas 
    em tempo real utilizando YOLOv8 em CPU.
    """
    # 1. Carrega o modelo YOLOv8 pré-treinado (versão nano para otimizar a CPU)
    print("Carregando o modelo YOLOv8...")
    model = YOLO("yolov8n.pt")
    
    # 2. Inicializa a webcam (0 geralmente indica a câmera padrão)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a webcam.")
        return

    # Opcional: Ajustar resolução para melhorar a fluidez em CPU (ex: 640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Iniciando transmissão. Pressione 'q' para sair.")

    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            print("Erro ao capturar o frame da câmera.")
            break

        # 3. Executa a inferência no frame atual (forçando CPU)
        # verbose=False oculta logs repetitivos no terminal a cada frame
        resultados = model(frame, device="cpu", verbose=False)
        
        # 4. Desenha as detecções (bounding boxes, rótulos e confiança) no frame
        frame_anotado = resultados[0].plot()

        # 5. Exibe o vídeo em uma janela do OpenCV
        cv2.imshow("Deteccao em Tempo Real - YOLOv8 (CPU)", frame_anotado)

        # Interrompe o loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. Libera os recursos da câmera e fecha as janelas
    cap.release()
    cv2.destroyAllWindows()
    print("Encerrado com sucesso.")

if __name__ == "__main__":
    iniciar_deteccao_webcam()





# import cv2
# from ultralytics import YOLO

# def detectar_pessoas(caminho_imagem, caminho_saida="resultado_deteccao.jpg", conf_minima=0.5):
#     """
#     Detecta pessoas em uma imagem utilizando YOLOv8 em CPU.
    
#     Args:
#         caminho_imagem (str): Caminho para a imagem de entrada.
#         caminho_saida (str): Caminho para salvar a imagem processada.
#         conf_minima (float): Limiar de confiança para filtragem (0 a 1).
#     """
#     # 1. Carrega o modelo YOLOv8 pré-treinado (versão nano para CPU)
#     model = YOLO("yolov8n.pt")
    
#     # 2. Carrega a imagem com o OpenCV
#     imagem = cv2.imread(caminho_imagem)
#     if imagem is None:
#         raise FileNotFoundError(f"Não foi possível carregar a imagem em: {caminho_imagem}")
    
#     # 3. Executa a inferência forçando o uso da CPU
#     resultados = model(imagem, conf=conf_minima, device="cpu")
    
#     # 4. Processa os resultados e desenha as marcações na imagem
#     for resultado in resultados:
#         boxes = resultado.boxes
#         for box in boxes:
#             # Identifica a classe detectada (0 no COCO = 'person')
#             id_classe = int(box.cls[0])
#             nome_classe = model.names[id_classe]
            
#             if nome_classe == "person":
#                 # Extrai as coordenadas da caixa delimitadora
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 confianca = float(box.conf[0])
                
#                 # Desenha o retângulo ao redor da pessoa
#                 cv2.rectangle(imagem, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
#                 # Adiciona o rótulo com a taxa de confiança
#                 texto = f"Pessoa: {confianca:.2f}"
#                 cv2.putText(imagem, texto, (x1, y1 - 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
#     # 5. Salva e exibe o resultado
#     cv2.imwrite(caminho_saida, imagem)
#     print('Imagem processada com sucesso e salva em: {caminho_saida}')

# # Exemplo de uso:
# detectar_pessoas("c.png")

