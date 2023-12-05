from typing import List

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.security import HTTPBearer, OAuth2PasswordBearer
from pydantic import BaseModel

from service.api.exceptions import ModelNotFoundError, UnauthorizedUserError, UserNotFoundError
from service.log import app_logger


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
router, bearer = APIRouter(), HTTPBearer()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={
        404: {
            "description": "Incorrect User or Model",
            "content": {"application/json": {"example": {"detail": "model_name or user_id not found"}}},
        },
        401: {
            "description": "Incorrect authorization token",
            "content": {"application/json": {"example": {"detail": "Authorization failed"}}},
        },
    },
)
async def get_reco(request: Request, model_name: str, user_id: int, token=Depends(bearer)) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}")
    app_logger.info(f"Request for user: {user_id}")

    if request.app.state.token != token.credentials:
        raise UnauthorizedUserError()

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    if model_name != "some_model":
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    k_recs = request.app.state.k_recs
    reco = [42 + i for i in range(k_recs)]

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
