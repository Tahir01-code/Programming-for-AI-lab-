import re
from pathlib import Path

p = Path(r"c:\Users\tahir\Desktop\task 5\task_5 (1).py")
text = p.read_text()
text = re.sub(r'r"X-X-Everywhere\.jpg"', 'image_path', text)
p.write_text(text)
print("replacement done")
