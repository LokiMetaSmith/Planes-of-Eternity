use std::collections::HashMap;

pub struct CFG {
    rules: HashMap<String, Vec<Vec<String>>>,
    start_symbol: String,
}

impl CFG {
    pub fn new(start_symbol: &str) -> Self {
        Self {
            rules: HashMap::new(),
            start_symbol: start_symbol.to_string(),
        }
    }

    pub fn add_rule(&mut self, non_terminal: &str, production: Vec<&str>) {
        let prod_owned = production.iter().map(|s| s.to_string()).collect();
        self.rules.entry(non_terminal.to_string()).or_insert_with(Vec::new).push(prod_owned);
    }

    pub fn evaluate(&self, max_depth: usize, mut seed: u32) -> Vec<String> {
        self.expand(&self.start_symbol, max_depth, &mut seed)
    }

    fn expand(&self, symbol: &str, depth: usize, seed: &mut u32) -> Vec<String> {
        if depth == 0 {
            return vec![];
        }

        if let Some(productions) = self.rules.get(symbol) {
            // Pseudo-random choice based on seed
            *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let idx = (*seed as usize) % productions.len();
            let chosen_prod = &productions[idx];

            let mut result = Vec::new();
            for sym in chosen_prod {
                if self.rules.contains_key(sym) {
                    result.extend(self.expand(sym, depth - 1, seed));
                } else {
                    // It's a terminal
                    result.push(sym.clone());
                }
            }
            result
        } else {
            // It's a terminal
            vec![symbol.to_string()]
        }
    }
}
