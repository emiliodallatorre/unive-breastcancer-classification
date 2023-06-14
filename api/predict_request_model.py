from pandas import DataFrame
from pydantic import BaseModel


class PredictRequestModel(BaseModel):
    radius: float
    perimeter: float
    area: float

    def to_dataframe(self) -> DataFrame:
        return DataFrame([[self.radius, self.perimeter, self.area]], columns=['Radius', 'Perimeter', 'Area'])
