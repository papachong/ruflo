//! Oracle pane — ASK verb routes to ruflo-neural-trader (stub for Phase B).

use aperture_swarm::{reply, Agent, Envelope};
use serde_json::{json, Value};

use crate::agent_runner::{symbol_of, verb};

pub struct OraclePane {
    id: &'static str,
    focus: Option<String>,
}

impl OraclePane {
    pub fn new() -> Self {
        Self {
            id: "aperture:pane.oracle",
            focus: None,
        }
    }
}

impl Agent for OraclePane {
    fn id(&self) -> &str {
        &self.id
    }

    async fn handle(&mut self, env: Envelope) -> Vec<Envelope> {
        match verb(&env) {
            Some("ASK") => {
                let prompt = env
                    .payload
                    .get("prompt")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                // TODO(phase-c): forward prompt to ruflo-neural-trader via
                // the swarm bus instead of returning a stub response.
                let answer = format!(
                    "(stub) received prompt of {} chars; will route to ruflo-neural-trader in Phase C",
                    prompt.len()
                );
                vec![reply(
                    &env,
                    json!({
                        "verb": "ASK.RESULT",
                        "prompt": prompt,
                        "answer": answer,
                        "focus": self.focus,
                    }),
                )]
            }
            Some("FOCUS") => {
                self.focus = symbol_of(&env);
                vec![]
            }
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_runner::panes::test_helpers::req;
    use serde_json::json;

    #[tokio::test]
    async fn echoes_ask() {
        let mut p = OraclePane::new();
        let outs = p
            .handle(req("ASK", json!({"prompt": "what is going on?"})))
            .await;
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].payload["verb"], "ASK.RESULT");
    }
}
