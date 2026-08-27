//! Explicit tokenizer profile orchestration.

#![allow(
    clippy::enum_variant_names,
    clippy::derive_partial_eq_without_eq,
    reason = "the generated SML event enum contains borrowed runtime payloads"
)]

use core::cell::Cell;

use sml::sml;

use super::event::{Defaults, Error, Model, PreId, Resolve, Resolved};

#[derive(Clone, Copy, Debug)]
struct Working {
    model: Model,
    defaults: Defaults,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ResolveRuntime<'dispatch> {
    model: &'dispatch str,
    pre: &'dispatch str,
    working: &'dispatch Cell<Working>,
    result: &'dispatch Cell<Result<Resolved, Error>>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Context;

const DEFAULTS_LLAMA: Defaults = Defaults {
    bos_id: 1,
    eos_id: 2,
    unk_id: 0,
    flags: Defaults::ESCAPE_WHITESPACES | Defaults::ADD_BOS | Defaults::ADD_SPACE_PREFIX,
    ..Defaults::SOURCE
};
const DEFAULTS_BERT: Defaults = Defaults {
    bos_id: 101,
    unk_id: 100,
    sep_id: 102,
    pad_id: 0,
    mask_id: 103,
    flags: Defaults::ESCAPE_WHITESPACES | Defaults::ADD_BOS | Defaults::ADD_SEP,
    ..Defaults::SOURCE
};
const DEFAULTS_GPT2: Defaults = Defaults {
    bos_id: 11,
    eos_id: 11,
    ..Defaults::SOURCE
};
const DEFAULTS_T5: Defaults = Defaults {
    eos_id: 1,
    unk_id: 2,
    pad_id: 0,
    ..Defaults::SOURCE
};
const DEFAULTS_PLAMO2: Defaults = Defaults {
    bos_id: 1,
    eos_id: 2,
    unk_id: 0,
    pad_id: 3,
    ..Defaults::SOURCE
};

sml! {
    ProfileResolver {
        // Model classification and source defaults.
        "state_model_decision"_s <= *"state_ready"_s + Resolve(ResolveRuntime<'dispatch>),
        "state_pre_decision"_s <= "state_model_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_model_none] / effect_model_none,
        "state_pre_decision"_s <= "state_model_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_model_llama] / effect_model_llama,
        "state_pre_decision"_s <= "state_model_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_model_gemma4] / effect_model_gemma4,
        "state_pre_decision"_s <= "state_model_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_model_gpt2] / effect_model_gpt2,
        "state_pre_decision"_s <= "state_model_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_model_bert] / effect_model_bert,
        "state_pre_decision"_s <= "state_model_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_model_t5] / effect_model_t5,
        "state_pre_decision"_s <= "state_model_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_model_rwkv] / effect_model_rwkv,
        "state_pre_decision"_s <= "state_model_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_model_plamo2] / effect_model_plamo2,
        "state_pre_decision"_s <= "state_model_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_model_unknown] / effect_model_unknown,

        // Pre-tokenizer classification and exact default composition.
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_default] / effect_pre_default,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_llama3] / effect_pre_llama3,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_jais2] / effect_pre_jais2,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_dbrx] / effect_pre_dbrx,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_smaug] / effect_pre_smaug,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_deepseek_llm] / effect_pre_deepseek_llm,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_deepseek_coder] / effect_pre_deepseek_coder,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_deepseek3_llm] / effect_pre_deepseek3_llm,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_youtu] / effect_pre_youtu,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_falcon] / effect_pre_falcon,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_mpt] / effect_pre_mpt,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_starcoder] / effect_pre_starcoder,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_gpt2] / effect_pre_gpt2,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_jais] / effect_pre_jais,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_refact] / effect_pre_refact,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_command_r] / effect_pre_command_r,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_qwen2] / effect_pre_qwen2,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_qwen35] / effect_pre_qwen35,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_stablelm2] / effect_pre_stablelm2,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_olmo] / effect_pre_olmo,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_poro] / effect_pre_poro,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_chatglm4] / effect_pre_chatglm4,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_viking] / effect_pre_viking,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_tekken] / effect_pre_tekken,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_smollm] / effect_pre_smollm,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_codeshell] / effect_pre_codeshell,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_bloom] / effect_pre_bloom,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_gpt3_finnish] / effect_pre_gpt3_finnish,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_exaone] / effect_pre_exaone,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_exaone4] / effect_pre_exaone4,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_exaone_moe] / effect_pre_exaone_moe,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_chameleon] / effect_pre_chameleon,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_minerva] / effect_pre_minerva,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_megrez] / effect_pre_megrez,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_gpt4o] / effect_pre_gpt4o,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_tiny_aya] / effect_pre_tiny_aya,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_superbpe] / effect_pre_superbpe,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_trillion] / effect_pre_trillion,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_granite_docling] / effect_pre_granite_docling,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_bailingmoe] / effect_pre_bailingmoe,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_seed_coder] / effect_pre_seed_coder,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_hunyuan] / effect_pre_hunyuan,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_hunyuan_dense] / effect_pre_hunyuan_dense,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_joyai_llm] / effect_pre_joyai_llm,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_kimi_k2] / effect_pre_kimi_k2,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_grok_2] / effect_pre_grok_2,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_afmoe] / effect_pre_afmoe,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_minimax_m2] / effect_pre_minimax_m2,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_solar_open] / effect_pre_solar_open,
        "state_done"_s <= "state_pre_decision"_s + completion<Resolve>(ResolveRuntime<'dispatch>) [guard_pre_unknown] / effect_pre_unknown,
        "state_ready"_s <= "state_done"_s + completion<Resolve>(ResolveRuntime<'dispatch>),

        // Every externally reachable state fails closed for unexpected input.
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_model_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_pre_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_done"_s + unexpected_event<_> / effect_unexpected,
    }
}

impl ProfileResolverStateMachine<Context> {
    pub(super) fn resolve(&mut self, event: Resolve<'_>) -> Result<Resolved, Error> {
        let working = Cell::new(Working {
            model: Model::Unknown,
            defaults: Defaults::SOURCE,
        });
        let result = Cell::new(Err(Error::Internal));
        let runtime = ResolveRuntime {
            model: event.model,
            pre: event.pre,
            working: &working,
            result: &result,
        };
        let _dispatch = self.process_event(ProfileResolverEvents::Resolve(runtime));
        result.get()
    }
}

impl ProfileResolverStateMachineContext for Context {
    fn guard_model_none(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.model, "none" | "no_vocab"))
    }
    fn guard_model_llama(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(event.model == "llama")
    }
    fn guard_model_gemma4(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(event.model == "gemma4")
    }
    fn guard_model_gpt2(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(event.model == "gpt2")
    }
    fn guard_model_bert(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(event.model == "bert")
    }
    fn guard_model_t5(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(event.model == "t5")
    }
    fn guard_model_rwkv(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(event.model == "rwkv")
    }
    fn guard_model_plamo2(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(event.model == "plamo2")
    }
    fn guard_model_unknown(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(!((matches!(event.model, "none" | "no_vocab"))
            || (event.model == "llama")
            || (event.model == "gemma4")
            || (event.model == "gpt2")
            || (event.model == "bert")
            || (event.model == "t5")
            || (event.model == "rwkv")
            || (event.model == "plamo2")))
    }

    fn effect_model_none(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        set_model(event, Model::None, Defaults::SOURCE);
        Ok(())
    }
    fn effect_model_llama(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        set_model(event, Model::SentencePiece, DEFAULTS_LLAMA);
        Ok(())
    }
    fn effect_model_gemma4(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        set_model(event, Model::SentencePiece, Defaults::SOURCE);
        Ok(())
    }
    fn effect_model_gpt2(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        set_model(event, Model::Bpe, DEFAULTS_GPT2);
        Ok(())
    }
    fn effect_model_bert(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        set_model(event, Model::WordPiece, DEFAULTS_BERT);
        Ok(())
    }
    fn effect_model_t5(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        set_model(event, Model::Unigram, DEFAULTS_T5);
        Ok(())
    }
    fn effect_model_rwkv(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        set_model(event, Model::Rwkv, Defaults::SOURCE);
        Ok(())
    }
    fn effect_model_plamo2(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        set_model(event, Model::Plamo2, DEFAULTS_PLAMO2);
        Ok(())
    }
    fn effect_model_unknown(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        set_model(event, Model::Unknown, Defaults::SOURCE);
        Ok(())
    }

    fn guard_pre_default(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "" | "default"))
    }
    fn guard_pre_llama3(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(
            event.pre,
            "llama3"
                | "llama-v3"
                | "llama-bpe"
                | "falcon3"
                | "falcon-h1"
                | "pixtral"
                | "midm-2.0"
                | "lfm2"
                | "jina-v5-nano"
        ))
    }
    fn guard_pre_jais2(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "jais2"))
    }
    fn guard_pre_dbrx(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "dbrx"))
    }
    fn guard_pre_smaug(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "smaug"))
    }
    fn guard_pre_deepseek_llm(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "deepseek-llm"))
    }
    fn guard_pre_deepseek_coder(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "deepseek-coder"))
    }
    fn guard_pre_deepseek3_llm(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "deepseek-v3"))
    }
    fn guard_pre_youtu(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "youtu"))
    }
    fn guard_pre_falcon(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "falcon"))
    }
    fn guard_pre_mpt(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "mpt"))
    }
    fn guard_pre_starcoder(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "starcoder"))
    }
    fn guard_pre_gpt2(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "gpt2" | "gpt-2"))
    }
    fn guard_pre_jais(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "jais"))
    }
    fn guard_pre_refact(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "refact"))
    }
    fn guard_pre_command_r(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "command-r"))
    }
    fn guard_pre_qwen2(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "qwen2"))
    }
    fn guard_pre_qwen35(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "qwen2.5" | "qwen35"))
    }
    fn guard_pre_stablelm2(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "stablelm2"))
    }
    fn guard_pre_olmo(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "olmo"))
    }
    fn guard_pre_poro(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "poro"))
    }
    fn guard_pre_chatglm4(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "chatglm4"))
    }
    fn guard_pre_viking(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "viking"))
    }
    fn guard_pre_tekken(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "tekken"))
    }
    fn guard_pre_smollm(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "smollm"))
    }
    fn guard_pre_codeshell(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "codeshell"))
    }
    fn guard_pre_bloom(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "bloom"))
    }
    fn guard_pre_gpt3_finnish(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "gpt3-finnish"))
    }
    fn guard_pre_exaone(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "exaone"))
    }
    fn guard_pre_exaone4(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "exaone4"))
    }
    fn guard_pre_exaone_moe(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "exaone-moe"))
    }
    fn guard_pre_chameleon(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "chameleon"))
    }
    fn guard_pre_minerva(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "minerva"))
    }
    fn guard_pre_megrez(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "megrez"))
    }
    fn guard_pre_gpt4o(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "gpt4o" | "gpt-4o"))
    }
    fn guard_pre_tiny_aya(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "tiny-aya"))
    }
    fn guard_pre_superbpe(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "superbpe"))
    }
    fn guard_pre_trillion(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "trillion"))
    }
    fn guard_pre_granite_docling(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "granite-docling"))
    }
    fn guard_pre_bailingmoe(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "bailingmoe"))
    }
    fn guard_pre_seed_coder(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "seed-coder"))
    }
    fn guard_pre_hunyuan(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "hunyuan"))
    }
    fn guard_pre_hunyuan_dense(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "hunyuan-dense"))
    }
    fn guard_pre_joyai_llm(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "joyai-llm"))
    }
    fn guard_pre_kimi_k2(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "kimi-k2"))
    }
    fn guard_pre_grok_2(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "grok-2"))
    }
    fn guard_pre_afmoe(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "afmoe"))
    }
    fn guard_pre_minimax_m2(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "minimax-m2"))
    }
    fn guard_pre_solar_open(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.pre, "solar-open"))
    }
    fn guard_pre_unknown(&self, event: &ResolveRuntime<'_>) -> Result<bool, ()> {
        Ok(!matches!(
            event.pre,
            "" | "default"
                | "llama3"
                | "llama-v3"
                | "llama-bpe"
                | "falcon3"
                | "falcon-h1"
                | "pixtral"
                | "midm-2.0"
                | "lfm2"
                | "jina-v5-nano"
                | "jais2"
                | "dbrx"
                | "smaug"
                | "deepseek-llm"
                | "deepseek-coder"
                | "deepseek-v3"
                | "youtu"
                | "falcon"
                | "mpt"
                | "starcoder"
                | "gpt2"
                | "gpt-2"
                | "jais"
                | "refact"
                | "command-r"
                | "qwen2"
                | "qwen2.5"
                | "qwen35"
                | "stablelm2"
                | "olmo"
                | "poro"
                | "chatglm4"
                | "viking"
                | "tekken"
                | "smollm"
                | "codeshell"
                | "bloom"
                | "gpt3-finnish"
                | "exaone"
                | "exaone4"
                | "exaone-moe"
                | "chameleon"
                | "minerva"
                | "megrez"
                | "gpt4o"
                | "gpt-4o"
                | "tiny-aya"
                | "superbpe"
                | "trillion"
                | "granite-docling"
                | "bailingmoe"
                | "seed-coder"
                | "hunyuan"
                | "hunyuan-dense"
                | "joyai-llm"
                | "kimi-k2"
                | "grok-2"
                | "afmoe"
                | "minimax-m2"
                | "solar-open"
        ))
    }

    fn effect_pre_default(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(0), defaults);
        Ok(())
    }
    fn effect_pre_llama3(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let mut defaults = event.working.get().defaults;
        defaults.flags |= Defaults::IGNORE_MERGES | Defaults::ADD_BOS;
        finish_pre(event, PreId(1), defaults);
        Ok(())
    }
    fn effect_pre_jais2(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(2), defaults);
        Ok(())
    }
    fn effect_pre_dbrx(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(3), defaults);
        Ok(())
    }
    fn effect_pre_smaug(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(4), defaults);
        Ok(())
    }
    fn effect_pre_deepseek_llm(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(5), defaults);
        Ok(())
    }
    fn effect_pre_deepseek_coder(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(6), defaults);
        Ok(())
    }
    fn effect_pre_deepseek3_llm(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(7), defaults);
        Ok(())
    }
    fn effect_pre_youtu(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let mut defaults = event.working.get().defaults;
        defaults.flags |= Defaults::IGNORE_MERGES;
        finish_pre(event, PreId(8), defaults);
        Ok(())
    }
    fn effect_pre_falcon(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(9), defaults);
        Ok(())
    }
    fn effect_pre_mpt(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(10), defaults);
        Ok(())
    }
    fn effect_pre_starcoder(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(11), defaults);
        Ok(())
    }
    fn effect_pre_gpt2(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(12), defaults);
        Ok(())
    }
    fn effect_pre_jais(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(13), defaults);
        Ok(())
    }
    fn effect_pre_refact(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(14), defaults);
        Ok(())
    }
    fn effect_pre_command_r(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(15), defaults);
        Ok(())
    }
    fn effect_pre_qwen2(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(16), defaults);
        Ok(())
    }
    fn effect_pre_qwen35(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(17), defaults);
        Ok(())
    }
    fn effect_pre_stablelm2(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(18), defaults);
        Ok(())
    }
    fn effect_pre_olmo(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(19), defaults);
        Ok(())
    }
    fn effect_pre_poro(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(20), defaults);
        Ok(())
    }
    fn effect_pre_chatglm4(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(21), defaults);
        Ok(())
    }
    fn effect_pre_viking(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(22), defaults);
        Ok(())
    }
    fn effect_pre_tekken(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(23), defaults);
        Ok(())
    }
    fn effect_pre_smollm(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(24), defaults);
        Ok(())
    }
    fn effect_pre_codeshell(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(25), defaults);
        Ok(())
    }
    fn effect_pre_bloom(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(26), defaults);
        Ok(())
    }
    fn effect_pre_gpt3_finnish(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(27), defaults);
        Ok(())
    }
    fn effect_pre_exaone(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(28), defaults);
        Ok(())
    }
    fn effect_pre_exaone4(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(29), defaults);
        Ok(())
    }
    fn effect_pre_exaone_moe(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(30), defaults);
        Ok(())
    }
    fn effect_pre_chameleon(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(31), defaults);
        Ok(())
    }
    fn effect_pre_minerva(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(32), defaults);
        Ok(())
    }
    fn effect_pre_megrez(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(33), defaults);
        Ok(())
    }
    fn effect_pre_gpt4o(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(34), defaults);
        Ok(())
    }
    fn effect_pre_tiny_aya(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(35), defaults);
        Ok(())
    }
    fn effect_pre_superbpe(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(36), defaults);
        Ok(())
    }
    fn effect_pre_trillion(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(37), defaults);
        Ok(())
    }
    fn effect_pre_granite_docling(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(38), defaults);
        Ok(())
    }
    fn effect_pre_bailingmoe(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(39), defaults);
        Ok(())
    }
    fn effect_pre_seed_coder(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(40), defaults);
        Ok(())
    }
    fn effect_pre_hunyuan(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(41), defaults);
        Ok(())
    }
    fn effect_pre_hunyuan_dense(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(42), defaults);
        Ok(())
    }
    fn effect_pre_joyai_llm(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(43), defaults);
        Ok(())
    }
    fn effect_pre_kimi_k2(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(44), defaults);
        Ok(())
    }
    fn effect_pre_grok_2(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(45), defaults);
        Ok(())
    }
    fn effect_pre_afmoe(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(46), defaults);
        Ok(())
    }
    fn effect_pre_minimax_m2(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(47), defaults);
        Ok(())
    }
    fn effect_pre_solar_open(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(48), defaults);
        Ok(())
    }
    fn effect_pre_unknown(&mut self, event: ResolveRuntime<'_>) -> Result<(), ()> {
        let defaults = event.working.get().defaults;
        finish_pre(event, PreId(49), defaults);
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        std::process::abort()
    }
}

fn set_model(event: ResolveRuntime<'_>, model: Model, defaults: Defaults) {
    event.working.set(Working { model, defaults });
}

fn finish_pre(event: ResolveRuntime<'_>, pre: PreId, defaults: Defaults) {
    event.result.set(Ok(Resolved {
        model: event.working.get().model,
        pre,
        defaults,
    }));
}

#[cfg(test)]
pub(super) const KNOWN_PRE_INPUT_COUNT: usize = 61;
