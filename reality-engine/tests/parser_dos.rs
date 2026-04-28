use reality_engine::lambda::parse;

#[test]
fn test_parser_dos() {
    let mut payload = String::new();
    for _ in 0..10000 {
        payload.push_str("(");
    }
    payload.push_str("FIRE");
    for _ in 0..10000 {
        payload.push_str(")");
    }

    // This should not crash the test suite
    parse(&payload);
}
