use std::fmt;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Primitive {
    Fire,
    Water,
    Earth,
    Air,
    Growth,
    Decay,
    Energy,
    Stable,
    Void,
}

impl fmt::Display for Primitive {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Primitive::Fire => write!(f, "FIRE"),
            Primitive::Water => write!(f, "WATER"),
            Primitive::Earth => write!(f, "EARTH"),
            Primitive::Air => write!(f, "AIR"),
            Primitive::Growth => write!(f, "GROWTH"),
            Primitive::Decay => write!(f, "DECAY"),
            Primitive::Energy => write!(f, "ENERGY"),
            Primitive::Stable => write!(f, "STABLE"),
            Primitive::Void => write!(f, "VOID"),
        }
    }
}

impl Primitive {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "FIRE" => Some(Primitive::Fire),
            "WATER" => Some(Primitive::Water),
            "EARTH" => Some(Primitive::Earth),
            "AIR" => Some(Primitive::Air),
            "GROWTH" => Some(Primitive::Growth),
            "DECAY" => Some(Primitive::Decay),
            "ENERGY" => Some(Primitive::Energy),
            "STABLE" => Some(Primitive::Stable),
            "VOID" => Some(Primitive::Void),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Term {
    Var(String),
    Abs(String, Rc<Term>), // \x. Body
    App(Rc<Term>, Rc<Term>), // (Function Argument)
    Prim(Primitive),
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Term::Var(name) => write!(f, "{}", name),
            Term::Abs(param, body) => write!(f, "(λ{}.{})", param, body),
            Term::App(func, arg) => write!(f, "({} {})", func, arg),
            Term::Prim(p) => write!(f, "{}", p),
        }
    }
}

impl Term {
    pub fn var(name: &str) -> Rc<Term> {
        Rc::new(Term::Var(name.to_string()))
    }

    pub fn abs(param: &str, body: Rc<Term>) -> Rc<Term> {
        Rc::new(Term::Abs(param.to_string(), body))
    }

    pub fn app(func: Rc<Term>, arg: Rc<Term>) -> Rc<Term> {
        Rc::new(Term::App(func, arg))
    }

    pub fn prim(p: Primitive) -> Rc<Term> {
        Rc::new(Term::Prim(p))
    }

    // Capture-avoiding substitution: self[val/var]
    pub fn substitute(&self, var_name: &str, val: &Rc<Term>) -> Rc<Term> {
        match self {
            Term::Var(name) => {
                if name == var_name {
                    val.clone()
                } else {
                    Rc::new(Term::Var(name.clone()))
                }
            }
            Term::App(func, arg) => {
                Rc::new(Term::App(
                    func.substitute(var_name, val),
                    arg.substitute(var_name, val)
                ))
            }
            Term::Abs(param, body) => {
                if param == var_name {
                    // Variable shadowed, stop substitution
                    Rc::new(Term::Abs(param.clone(), body.clone()))
                } else if val.free_vars().contains(param) {
                    // Capture risk! Rename bound variable.
                    let new_param = format!("{}'", param);
                    let new_body = body.substitute(param, &Term::var(&new_param));
                    Rc::new(Term::Abs(
                        new_param.clone(),
                        new_body.substitute(var_name, val)
                    ))
                } else {
                    Rc::new(Term::Abs(
                        param.clone(),
                        body.substitute(var_name, val)
                    ))
                }
            }
            Term::Prim(p) => Rc::new(Term::Prim(*p)),
        }
    }

    pub fn free_vars(&self) -> Vec<String> {
        match self {
            Term::Var(name) => vec![name.clone()],
            Term::App(func, arg) => {
                let mut vars = func.free_vars();
                vars.extend(arg.free_vars());
                vars.sort();
                vars.dedup();
                vars
            }
            Term::Abs(param, body) => {
                let mut vars = body.free_vars();
                vars.retain(|v| v != param);
                vars
            }
            Term::Prim(_) => vec![],
        }
    }

    // Normal Order Reduction (reduce leftmost outermost redex)
    // Returns (NewTerm, DidReduce)
    pub fn reduce(&self) -> (Rc<Term>, bool) {
        match self {
            Term::App(func, arg) => {
                // Try to reduce the function part first (Normal Order strategy requires checking if func is an Abstraction)

                // If func is an abstraction, we BETA REDUCE!
                if let Term::Abs(param, body) = &**func {
                     // (\x.M) N -> M[N/x]
                     return (body.substitute(param, arg), true);
                }

                // If func is NOT an abstraction yet, try to reduce it.
                let (new_func, reduced_func) = func.reduce();
                if reduced_func {
                    return (Term::app(new_func, arg.clone()), true);
                }

                // If func couldn't reduce, try reducing the argument
                let (new_arg, reduced_arg) = arg.reduce();
                if reduced_arg {
                    return (Term::app(func.clone(), new_arg), true);
                }

                // Nothing to reduce
                (Rc::new(self.clone()), false)
            }
            Term::Abs(param, body) => {
                // Reduce under abstraction? (Full Normal Order)
                let (new_body, reduced) = body.reduce();
                if reduced {
                    (Term::abs(param, new_body), true)
                } else {
                    (Rc::new(self.clone()), false)
                }
            }
            _ => (Rc::new(self.clone()), false),
        }
    }
}

// Simple Parser
pub fn parse(input: &str) -> Option<Rc<Term>> {
    let tokens = tokenize(input);
    let (term, _) = parse_tokens(&tokens)?;
    Some(term)
}

#[derive(Debug, PartialEq, Clone)]
enum Token {
    Lambda,
    Dot,
    LParen,
    RParen,
    Ident(String),
}

fn tokenize(input: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&c) = chars.peek() {
        match c {
            '\\' | 'λ' => {
                tokens.push(Token::Lambda);
                chars.next();
            }
            '.' => {
                tokens.push(Token::Dot);
                chars.next();
            }
            '(' => {
                tokens.push(Token::LParen);
                chars.next();
            }
            ')' => {
                tokens.push(Token::RParen);
                chars.next();
            }
            c if c.is_whitespace() => {
                chars.next();
            }
            c if c.is_alphanumeric() => {
                let mut ident = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_alphanumeric() || c == '_' {
                        ident.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Ident(ident));
            }
            _ => { chars.next(); } // Skip unknown
        }
    }
    tokens
}

fn parse_tokens(tokens: &[Token]) -> Option<(Rc<Term>, &[Token])> {
    if tokens.is_empty() { return None; }

    // Parse one term (atom or parenthesized or lambda)
    let (mut left, mut rest) = parse_atom(tokens)?;

    // Handle application: Term Term ...
    while let Some((right, next_rest)) = parse_atom(rest) {
        left = Term::app(left, right);
        rest = next_rest;
    }

    Some((left, rest))
}

fn parse_atom(tokens: &[Token]) -> Option<(Rc<Term>, &[Token])> {
    if tokens.is_empty() { return None; }

    match &tokens[0] {
        Token::LParen => {
            let (term, rest) = parse_tokens(&tokens[1..])?;
            if rest.first() == Some(&Token::RParen) {
                Some((term, &rest[1..]))
            } else {
                None // Unbalanced
            }
        }
        Token::Lambda => {
            // \x. Body
            if let Some(Token::Ident(param)) = tokens.get(1) {
                if tokens.get(2) == Some(&Token::Dot) {
                    let (body, rest) = parse_tokens(&tokens[3..])?;
                    Some((Term::abs(param, body), rest))
                } else {
                    None
                }
            } else {
                None
            }
        }
        Token::Ident(name) => {
            // Check if it's a Primitive
            if let Some(prim) = Primitive::from_str(name) {
                Some((Term::prim(prim), &tokens[1..]))
            } else {
                Some((Term::var(name), &tokens[1..]))
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_parse() {
        let term = parse("FIRE").unwrap();
        match *term {
            Term::Prim(Primitive::Fire) => assert!(true),
            _ => assert!(false, "Expected Prim(Fire)"),
        }
    }

    #[test]
    fn test_primitive_app() {
        let term = parse("GROWTH TREE").unwrap(); // TREE is var, GROWTH is prim
        // "TREE" is not a known prim, so it's a Var("TREE")
        println!("{}", term);
        assert_eq!(term.to_string(), "(GROWTH TREE)");
    }

    #[test]
    fn test_identity() {
        let term = parse("(\\x.x) y").unwrap();
        let (reduced, changed) = term.reduce();
        assert!(changed);
        assert_eq!(reduced.to_string(), "y");
    }

    #[test]
    fn test_complex() {
        // (\x.\y.x) a b -> a
        let term = parse("(\\x.\\y.x) a b").unwrap(); // ((\x.\y.x) a) b
        let (step1, _) = term.reduce(); // (\y.a) b
        let (step2, _) = step1.reduce(); // a
        assert_eq!(step2.to_string(), "a");
    }

    #[test]
    fn test_shadowing() {
        // (\x.(\x.x)) y -> (\x.x)
        // Outer x is replaced by y? No, inner x shadows it.
        // Body is (\x.x). substitute(x, y) on (\x.x) should return (\x.x)
        let term = parse("(\\x.(\\x.x)) y").unwrap();
        let (reduced, _) = term.reduce();
        assert_eq!(reduced.to_string(), "(λx.x)");
    }

    #[test]
    fn test_capture_avoidance() {
        // (\x.\y.x) y -> (\y'.y) NOT (\y.y)
        // x is substituted by y. y is bound in inner abs.
        let term = parse("(\\x.\\y.x) y").unwrap();
        let (reduced, _) = term.reduce();
        // Inner param 'y' should be renamed to avoid capturing the free 'y' being passed in.
        // The implementation appends '
        // Expect: (\y'.y)
        println!("Reduced: {}", reduced);
        assert!(reduced.to_string().contains("y'"));
    }
}
