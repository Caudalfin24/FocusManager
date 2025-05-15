import torch
from torchvision import transforms
from PIL import Image
import cv2
import mediapipe as mp
import numpy as np
from emotion_net import EmotionNet


class Scorer:
    def __init__(self, model_path):
        # 初始化情感识别模型
        self.emotion_labels =['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.emotion_scores = {'angry': 0, 'disgust': 0, 'fear': 50, 'happy': 100, 'neutral': 50, 'sad': 0, 'surprise': 100}
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmotionNet(num_classes=7)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        # 初始化人脸检测
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True)
    
    def predict_emotion(self, img) -> str:
        """情感识别

        Args:
            img (MatLike): 输入图像

        Returns:
            str: 识别的情感标签
        """
        image = Image.fromarray(img)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
        return self.emotion_labels[pred]
    
    def estimate_head_pose(self, img):
        """输入图像，进行头部姿态估计，返回 pitch、yaw、roll

        Args:
            img (MatLike): 输入的 OpenCV 人脸图像
        
        Returns:
            (float, float, float): 返回 pitch, yaw, roll
        """
        # rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        results = self.face_mesh.process(img)
        if not results.multi_face_landmarks:
            return None, None, None
        face_landmarks = results.multi_face_landmarks[0]
        img_h, img_w = img.shape[:2]
        
        # 获取关键点
        landmark_idxs = [1,152,33,263,61,291]
        image_points = []
        for idx in landmark_idxs:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            image_points.append((x,y))
        image_points = np.array(image_points, dtype='double')
        model_points = np.float32([
            [0.0, 0.0, 0.0],
            [0.0, -63.6, -12.5],
            [-43.3, 32.7, -26.0],
            [43.3, 32.7, -26.0],
            [-28.9, -28.9, -24.1],
            [28.9, -28.9, -24.1]
        ])
        
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )
        
        rmat, _ = cv2.Rodrigues(rotation_vector)
        pose_mat = cv2.hconcat((rmat, translation_vector))
        _,_,_,_,_,_,euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        pitch, yaw, roll = [float(a) for a in euler_angles]
        # 规约修正
        pitch = pitch + 180
        if pitch >= 180:
            pitch -= 360
        return pitch, yaw, roll
    
    def clip_face(self, img):
        """识别人脸并返回人脸图像

        Args:
            img (MatLike): 输入的 OpenCV 图像

        Returns:
            MatLike: 识别的人脸图像，若无，返回 None
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) == 0:
            print("Face Detect Missed.")
            return None
        (x,y,w,h) = faces[0]
        print("Face Detected:", x,y,w,h)
        face_img = img[y:y+h, x:x+w]
        return face_img
            
    def score(self, img, alpha = 0.3, beta=0.3, gamma=0.4):
        """输入图像及评分权重超参数，计算专注度评分

        Args:
            img (MatLike): 输入的 OpenCV 图像.
            alpha (float, optional): 基准评分权重. Defaults to 0.3.
            beta (float, optional): 头部姿态估计评分权重. Defaults to 0.3.
            gamma (float, optional): 情感识别评分权重. Defaults to 0.4.

        Returns:
            float: 评分
        """
        img = self.clip_face(img)
        if img is None:
            return 0.0
        
        # 头部姿态检测
        pitch, yaw, roll = self.estimate_head_pose(img)
        if pitch == None:
            return 0.0
        print("Head Position:", pitch, yaw, roll)
        head_score = - pitch - 7/3 * yaw + 95 
        if head_score < 0.0:
            head_score = 0.0
            
        # 情感检测
        emotion = self.predict_emotion(img)
        print('Predict Emotion:', emotion)
        emotion_score = self.emotion_scores[emotion]
        
        score = alpha * 100 + beta * head_score + gamma * emotion_score
        return score
    
if __name__ == "__main__":   
    model_path = "../model/emotion_net_best.pth"
    pic_path = "../model/data/FER2013/test/angry/PrivateTest_12766285.jpg"

    scorer = Scorer(model_path)
    img = cv2.imread(pic_path)
    score = scorer.score(img)
    print(score)