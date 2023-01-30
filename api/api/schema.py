from pydantic import BaseModel

class Text(BaseModel):
    text: str = "Eh, want to go out anot"