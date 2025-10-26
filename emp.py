class employee:
    def __init__(self,id,name):
        self.id=id
        self.name=name
    def display(self):
        print(f"Id: {self.id},name: {self.name}")

e1=employee(1,"Alice")
e1.display()
e2=employee(2,"Bob")    
e2.display()