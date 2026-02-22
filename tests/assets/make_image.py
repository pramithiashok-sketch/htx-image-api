from PIL import Image

img = Image.new("RGB", (100, 100), color="blue")
img.save("tests/assets/test.jpg")
print("made tests/assets/test.jpg")