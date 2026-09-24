use actix_web::{http::StatusCode, HttpResponse, ResponseError};
use std::fmt::{Display, Formatter, Result as FormatResult};

use crate::api::dtos::ErrorBody;

/// Domain error for the evaluation pipeline.
///
/// Each variant maps to an HTTP status code through [`ResponseError`]:
/// invalid requests are `422`, unknown models are `404` and inference
/// failures are server errors (`500`).
#[derive(Debug)]
pub enum EvaluationError {
    InvalidRequest(String),
    UnknownModel(String),
    Inference(String),
}

impl EvaluationError {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest(message.into())
    }

    pub fn unknown_model(message: impl Into<String>) -> Self {
        Self::UnknownModel(message.into())
    }

    pub fn inference(message: impl Into<String>) -> Self {
        Self::Inference(message.into())
    }

    fn message(&self) -> &str {
        match self {
            Self::InvalidRequest(message)
            | Self::UnknownModel(message)
            | Self::Inference(message) => message,
        }
    }
}

impl Display for EvaluationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FormatResult {
        match self {
            Self::InvalidRequest(message) => {
                write!(formatter, "invalid request: {message}")
            }
            Self::UnknownModel(message) => {
                write!(formatter, "unknown model: {message}")
            }
            Self::Inference(message) => {
                write!(formatter, "inference failure: {message}")
            }
        }
    }
}

impl std::error::Error for EvaluationError {}

impl From<candle_core::Error> for EvaluationError {
    fn from(error: candle_core::Error) -> Self {
        Self::Inference(error.to_string())
    }
}

impl From<anyhow::Error> for EvaluationError {
    fn from(error: anyhow::Error) -> Self {
        Self::Inference(error.to_string())
    }
}

impl ResponseError for EvaluationError {
    fn status_code(&self) -> StatusCode {
        match self {
            Self::InvalidRequest(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::UnknownModel(_) => StatusCode::NOT_FOUND,
            Self::Inference(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn error_response(&self) -> HttpResponse {
        HttpResponse::build(self.status_code()).json(ErrorBody::new(self.message()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_request_maps_to_unprocessable_entity() {
        let error = EvaluationError::invalid_request("questions must not be empty");
        assert_eq!(error.status_code(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn unknown_model_maps_to_not_found() {
        let error = EvaluationError::unknown_model("model 'ghost' is not served");
        assert_eq!(error.status_code(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn inference_failure_maps_to_internal_server_error() {
        let error = EvaluationError::inference("forward pass failed");
        assert_eq!(error.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn error_response_carries_standard_json_envelope() {
        let error = EvaluationError::unknown_model("model 'ghost' is not served");
        let response = error.error_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn display_prefixes_each_variant() {
        assert!(EvaluationError::invalid_request("bad")
            .to_string()
            .contains("bad"));
        assert!(EvaluationError::unknown_model("ghost")
            .to_string()
            .contains("ghost"));
        assert!(EvaluationError::inference("boom")
            .to_string()
            .contains("boom"));
    }

    #[test]
    fn candle_errors_convert_to_inference_failures() {
        let candle_error = candle_core::Error::Msg("weight mismatch".to_string());
        let error = EvaluationError::from(candle_error);
        assert_eq!(error.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
