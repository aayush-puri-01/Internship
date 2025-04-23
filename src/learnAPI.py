from fastapi import FastAPI, Path, Query
from typing import Optional, Union
from pydantic import BaseModel

app = FastAPI()

class StudentSchema(BaseModel):
    name : str
    address : str
    remarks : str
    age: int

class UpdateStudentSchema(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    remarks: Optional[str] = None
    age: Optional[int] = None

students = {
    1 : {
        "name": "John Doe",
        "address": "KisimNagar",
        "remarks": "Excellento",
        "age": 16
    },
    2 : {
        "name": "Savannah",
        "address": "Going",
        "remarks": "Swish",
        "age": 18
    }
}

#path parameter

@app.get("/get-students/{student_id}")
def get_student(student_id: int = Path(..., description = "Enter Student ID", gt=0, lt=5)):
    return students[student_id]

#query parameter

@app.get("/get-student-by-name")
def get_student_by_name(Name: str):
    for s in students:
        if students[s]["name"] == Name:
            return students[s]
    return {"Data": "Not Found"}

@app.post("/add-new-student/{student_id}")
def add_student(*, student_id: int = Path(description="New ID for New Student"), any_student: StudentSchema):
    if student_id in students:
        return {"Error": "Student Already Exists"}
    students[student_id] = any_student

@app.get("/get-student-ids")
def get_ids():
    available_keys = list(students.keys())
    return available_keys
print(students.keys())

@app.put("/update-student-info/{student_id}")
def update_studnet(*, student_id: int = Path(description="Id of the student to be updated"), patch_data: UpdateStudentSchema):
    if student_id not in students:
        return {"Error": "Student does not exist"}
    patch_dict = patch_data.model_dump(exclude_none=False)
    patch_list = list(patch_dict.keys())
    for k in patch_list:
        if patch_dict[k] != None:
            students[student_id][k] = patch_dict[k]

    return students[student_id]