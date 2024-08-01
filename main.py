import cv2
import face_recognition
import os

# Загрузим все изображения из папки photos в массив known_faces
known_faces = []
names = []  # массив имен людей на фото
for filename in os.listdir('photos'):
    img = face_recognition.load_image_file('photos/' + filename)
    face_encoding = face_recognition.face_encodings(img)[0]
    known_faces.append(face_encoding)
    names.append(os.path.splitext(filename)[0])

def identify_face(image_path):
    # Идентифицируем человека на фото
    unknown_image = face_recognition.load_image_file(image_path)

    # Уменьшаем изображение до 50% от исходного
    height, width = unknown_image.shape[:2]
    resized_image = cv2.resize(unknown_image, (int(width / 2), int(height / 2)))

    face_locations = face_recognition.face_locations(resized_image)
    face_encodings = face_recognition.face_encodings(resized_image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Сравниваем лицо с известными лицами
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        # Если нашли совпадение, выводим имя человека
        if True in matches:
            first_match_index = matches.index(True)
            name = names[first_match_index]

        # Отображаем результат на изображении
        cv2.rectangle(resized_image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(resized_image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

    # Отображаем изображение с результами
    cv2.imshow('Identified Face', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_path = input("Введите путь к фото: ")
    identify_face(image_path)