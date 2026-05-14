import cv2
import torch
from torchvision import transforms
from PIL import Image
from model import MobileViT_XXS
import winsound  # for the beep alarm

def run_live_inference():
    # setup device and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    # class 0 = drowsy, class 1 = alert
    model = MobileViT_XXS(num_classes=2)
    model.load_state_dict(torch.load("best_mobilevit_drowsiness.pth", map_location=device))
    model.to(device)
    model.eval()

    # same transforms as training
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # start webcam
    cap = cv2.VideoCapture(0)
    
    drowsy_frame_count = 0#change 1 in the following line
    ALARM_THRESHOLD = 12  # how many drowsy frames before alarm

    print("Starting webcam... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # detect faces using grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        # keep a clean frame for model input
        display_frame = frame.copy()

        for (x, y, w, h) in faces:
            # add a bit of padding around face
            pad_x = int(w * 0.20) 
            pad_y = int(h * 0.20) 
            
            y1 = max(0, y - pad_y)
            y2 = min(frame.shape[0], y + h + pad_y)
            x1 = max(0, x - pad_x)
            x2 = min(frame.shape[1], x + w + pad_x)

            # crop face only
            face_roi = frame[y1:y2, x1:x2]            
            
            
            
            # convert to PIL format
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)

            # preprocess and add batch dim
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            # make prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                
                # get confidence scores
                probs = torch.nn.functional.softmax(outputs, dim=1)
                prob_drowsy = probs[0][0].item()
                prob_alert = probs[0][1].item()
                
                # print confidence in terminal
                print(f"Confidence -> Drowsy: {prob_drowsy*100:.1f}% | Alert: {prob_alert*100:.1f}%")

                _, predicted = torch.max(outputs.data, 1)
                
                # decide if person is drowsy
                is_drowsy = (predicted.item() == 0) or (prob_drowsy > 0.45)#change 2

            # show what model is actually seeing
            cv2.imshow('What the Model Sees', face_roi)

            # draw face box
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            if is_drowsy:
                drowsy_frame_count += 1
                color = (0, 0, 255) # red
                label = "DROWSY"
                
                if drowsy_frame_count >= ALARM_THRESHOLD:
                    cv2.putText(display_frame, "WAKE UP!", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    # beep if drowsy too long
                    winsound.Beep(1000, 500) 
            else:
                drowsy_frame_count = 0
                color = (0, 255, 0) # green 
                label = "ALERT"

            # show status label
            cv2.putText(display_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # show webcam feed
        cv2.imshow('MobileViT Drowsiness Detection', display_frame)

        # exiting option
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cleanup stuff
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_inference()