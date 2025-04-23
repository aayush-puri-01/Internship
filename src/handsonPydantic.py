from pydantic import BaseModel, EmailStr, field_validator, model_validator, ValidationError
from typing import List
import json


@model_validator(mode = 'before')
@classmethod
def checkInputFormat(data):
    if not isinstance(data, dict):
        raise TypeError("Input data must be a dictionary")
    return data

@field_validator('age')
def checkAgeValidity(age):
    if age <= 30:
        raise ValueError("Bro how are you even a student at this point man")
    return age

class Student(BaseModel):
    name: str
    age: int
    address: str
    email: EmailStr
    standard: int


class StudentPortal:
    def __init__(self):
        self.students: List[Student] = []

    def add_student(self, data):
        try:
            student = Student.model_validate(data)
            self.students.append(student)
            print("Student Added Successfully")
        except Exception as e:
            print(f"Validation error: {e}")

    def list_students(self):
        for student in self.students:
            print(student.model_dump_json())

    def save_to_file(self, filename="students.json"):
        with open(filename, "w") as f:
            json.dump([s.model_dump() for s in self.students], f, indent=2)
        print("Saved to file Successfully")

    def load_from_file(self, filepath):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                self.students = [Student.model_validate(s) for s in data]
            print("Successfully loaded from file")
        except FileNotFoundError:
            print("No existing file, starting fresh")
        except ValidationError as e:
            print(f"File data is invalid: {e}")

if __name__ == "main":
    portal = StudentPortal()
    portal.load_from_file("file.txt")

    while True:
        print("------Student Portal------\n")
        print("1. Add student\n")
        print("2. List student\n")
        print("1. Save to file \n")
        print("4. Exit \n")

    choice = input("Enter your choice")
