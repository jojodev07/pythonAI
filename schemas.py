from pydantic import BaseModel, Field, field_validator

class QueryRequest(BaseModel):
    question : str
    top_k : int = Field(default=3, ge=1)

class QueryResponse(BaseModel):
    answer : str
    context : list[str]

class StageDetail(BaseModel):
    time_minutes: int = Field(description= "الزمن للمرحلة بالدقائق")
    teacher_action: str = Field(description="دور المعلم")
    learner_action: str = Field(description="دور المتعلم")
    
class LessonPlanStages(BaseModel):
    engagement: StageDetail = Field(description="التهيئة والاندماج")
    explanation: StageDetail = Field(description="الشرح والتفصيل مع مراعاة التعليم المتمايز")
    elaboration: StageDetail = Field(description="التوسع ودعم التميز")
    closing: StageDetail = Field(description="تأكيد التعلم / الغلق")
    
class LessonPlanSchema(BaseModel):
    subject: str = Field(description="المبحث")
    unit_title: str = Field(description="عنوان الوحدة")
    lesson_title: str = Field(description="موضوع الدرس")
    lessons_count: int = Field(description="عدد الحصص")
    prior_learning: str = Field(description="التعلم القبلي")
    learning_outcomes: list[str] = Field(description="النتاجات التعليمية بأفعال سلوكية وشرط قياس")
    stages: LessonPlanStages
    self_reflection: str = Field(description="أسئلة التأمل الذاتي للمعلم")
    
class LessonPlanRequest(BaseModel):
    subject: str = Field(..., example="التربية الوطنية والمدنية")
    unit_title: str = Field(..., example="الدولة ومؤسساتها")
    lesson_title: str = Field(..., example="الدولة")
    grade: str = Field(..., example="التاسع")
    periods: int = Field(default=2, example=2)