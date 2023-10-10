from torchinfo import summary

from arch.PISR.fsrcnn import FSRCNNTeacher, FSRCNNStudent

Teacher = FSRCNNTeacher(scale=2, n_colors=1)
Student = FSRCNNStudent(scale=2, n_colors=1)



summary(Teacher, (3, 128, 128), verbose=2, device='cpu', depth=3)

summary(Student, (3, 64, 64), verbose=2, device='cpu', depth=3)