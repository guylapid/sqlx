//! Parsing support for Generalized Query Placeholders, similar to `println!()` or `format_args!()` syntax.
//!
//! ### Kinds
//!
//! Implicit indexing: `SELECT * FROM foo WHERE id = {} AND bar = {}`
//! where each placeholder implicitly refers to an expression at the equivalent position
//! in the bind arguments list
//!
//! Explicit zero-based indexing: `SELECT * FROM foo WHERE id = {N}` where `N` is an unsigned integer
//! which refers to the Nth expression in the bind arguments list (starting from zero)
//!
//! Arguments capturing:
//!
//! `SELECT * FROM foo WHERE id = {<ident>}` where `<ident>` is a Rust identifier
//! defined in the same scope as the query string (for the macros) or an explicitly named bind argument
//! (for the dynamic interface)
//!
//! `SELECT * FROM foo WHERE id = {<field-expr>}` where `<field-expr>` is a Rust field expression
//! (e.g. `foo.bar.baz`) which resolves in the current scope (for the macros)
//!
//! Repetition interpolated into the query string:
//!
//! * `SELECT * FROM foo WHERE id IN ({+})`
//! * `SELECT * FROM foo WHERE id IN ({N+})`
//! * `SELECT * FROM foo WHERE id IN ({<ident>+})`
//! * `SELECT * FROM foo WHERE id IN ({(<field-expr>)+})`
//!
//! Similar to the above, but where the bind argument corresponding to the placeholder is expected
//! to be an iterable, and the repetition is expanded into the query string at runtime
//! (for databases which don't support binding arrays).
//!
//! For example:
//!
//! ```rust,ignore
//! let foo = [1, 2, 3, 4, 5];
//!
//! sqlx::query!("SELECT * FROM foo WHERE id IN ({foo*}")
//!
//! // would be equivalent to:
//!
//! sqlx::query!("SELECT * FROM foo WHERE id IN ($1, $2, $3, $4, $5)", foo[0], foo[1], foo[2], foo[3], foo[4])
//! ```
//!
//! (Note: for Postgres, binding the array directly instead of using expansion should be preferred
//! as it will not generate a different query string for every arity of iterable passed.)
//!
//! ### Potential Pitfalls to Avoid
//! We want to make sure to avoid trying to parse paired braces inside strings as it could
//! be, e.g. a JSON object literal. We also need to support escaping braces (and erasing the escapes)
//!

use std::borrow::Cow;
use std::fmt::{self, Display, Formatter, Write};
use std::ops::Range;

use crate::Database;
use combine::parser::char::{alpha_num, digit, letter};
use combine::parser::range::{recognize, recognize_with_value, take_while1};
use combine::parser::repeat::{escaped, repeat_skip_until};
use combine::stream::position::{Positioner, RangePositioner, SourcePosition};
use combine::*;

/// A query parsed for generic placeholders with [`parse_query()`].
pub struct ParsedQuery<'a> {
    pub(crate) query: &'a str,
    pub(crate) placeholders: Vec<Placeholder<'a>>,
}

/// A single generic placeholder in a query parsed with [`parse_query()`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Placeholder<'a> {
    /// The byte range in the source query where this placeholder appears, including the `{}`
    pub token: Range<usize>,
    /// The identifier for this placeholder.
    pub ident: Ident<'a>,
    /// The kleene operator for this placeholder. If `Some`, the bind parameter is expected to be a vector.
    pub kleene: Option<Kleene>,
}

/// The identifier for a placeholder which connects it to a bind parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Ident<'a> {
    /// An implicitly indexed placeholder, i.e. just `{}`
    Implicit,
    /// A positionally indexed placeholder, e.g. `{0}`, `{1}`, etc.
    Positional(u16),
    /// A named placeholder, e.g. `{foo}` would be `Named("foo")`
    Named(Cow<'a, str>),
    /// A placeholder with a field access expression, e.g. `{(foo.bar.baz)}` would be `Field("foo.bar.baz")`
    Field(Cow<'a, str>),
}

/// The optional Kleene operator of a [Placeholder] which changes its expansion.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Kleene {
    // not currently supported
    // Question,
    // Star,
    /// The `+` Kleene operator, e.g. `{foo+}`.
    ///
    /// A vector of 0 items expands to the literal `NULL` while
    /// a non-empty vector expands to a comma-separated list, e.g. `$1, $2, $3`.
    Plus,
}

/// The bind parameter indexing type for the given database.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParamIndexing {
    /// Implicitly indexed bind parameters, e.g. for MySQL
    /// which just does `SELECT 1 FROM foo WHERE bar = ? AND baz = ?`
    Implicit,
    /// Explicitly 1-based indexing of bind parameters, e.g. for Postgres
    /// which does `SELECT 1 FROM foo WHERE bar = $1 AND baz = $2`
    OneIndexed,
}

/// The type of an individual bind param.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BindParam {
    /// This bind param is a scalar, i.e. it should expand to only one concrete placeholder.
    Scalar,
    /// This bind param is a vector, i.e. its expansion is dictated by the `Kleene` operator.
    /// The `usize` value is the length of the vector (which may be 0).
    ///
    /// [`ParsedQuery::expand()`] will error if the corresponding [`Placeholder::kleene`] is `None`.
    Vector(usize),
}

/// The error type returned by [`parse_query`] and [`ParsedQuery::expand()`]
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// An error occurred while parsing the query for generic placeholder syntax.
    Parse {
        /// The byte position in the query string where the error occurred.
        byte_position: usize,
        /// The line in the string where the error occurred.
        line: i32,
        /// The column in the string where the error occurred.
        column: i32,
        /// The message string, with error and context.
        message: String,
        /// The context string, which may help with locating the error.
        context: String,
    },
    /// An error occurred while expanding the generic placeholder syntax
    Expand(String),
}

type Result<T, E = Error> = std::result::Result<T, E>;

impl<'a> ParsedQuery<'a> {
    /// Get the parsed list of placeholders.
    pub fn placeholders(&self) -> &[Placeholder<'a>] {
        &self.placeholders
    }

    /// Expand the placeholders in this query according to
    /// [`DB::PLACEHOLDER_CHAR`][Database::PLACEHOLDER_CHAR] and
    /// [`DB::PARAMETER_STYLE`][Database::PARAMETER_STYLE].
    ///
    /// The given iterator of [`BindParam`] must have the same length as [`Self::placeholders()`]
    /// where each element yielded by `params` corresponds 1:1 with a [`Placeholder`].
    ///
    /// Returns an error if:
    /// * `params.into_iter().count() != self.placeholders.len()`
    /// * any param is a [`BindParam::Scalar`] but the corresponding [`Placeholder::kleene`] is `Some`
    /// * any param is a [`BindParam::Vector`] but the corresponding [`Placeholder::kleene`] is `None`
    pub fn expand<DB: Database>(
        &self,
        params: impl IntoIterator<Item = BindParam>,
    ) -> Result<Cow<'a, str>> {
        self.expand_inner(DB::PLACEHOLDER_CHAR, DB::PARAM_INDEXING, params)
    }

    /// Unit-testable version of `expand`
    fn expand_inner(
        &self,
        placeholder_char: char,
        indexing: ParamIndexing,
        params: impl IntoIterator<Item = BindParam>,
    ) -> Result<Cow<'a, str>> {
        macro_rules! err {
            ($($args:tt)*) => {
                Err(Error::Expand(format!($($args)*)))
            };
        }

        // optimization: if we don't have any placeholders to substitute, then just return `self.query`
        if self.placeholders.is_empty() {
            let unused_params = params.into_iter().count();

            if unused_params != 0 {
                return err!("{} unused BindParams", unused_params);
            }

            return Ok(self.query.into());
        }

        // the current placeholder index; unused if `ParamIndexing::Implicit`
        let mut index = match indexing {
            ParamIndexing::Implicit => None,
            ParamIndexing::OneIndexed => Some(1),
        };

        let mut push_placeholder = |buf: &mut String| {
            buf.push(placeholder_char);

            if let Some(ref mut index) = index {
                write!(buf, "{}", index).expect("write!() to a string is infallible");
                *index += 1;
            }
        };

        let mut placeholders = self.placeholders.iter();
        let mut params = params.into_iter();
        let mut out = String::with_capacity(self.query.len());

        // copy `this .. self.query.len()` to the end of `out` after processing `placeholders`
        let mut last_placeholder_end = 0;

        for (placeholder, param) in placeholders.by_ref().zip(&mut params) {
            // push the chunk of `self.query` between the last placeholder and this one
            out.push_str(&self.query[last_placeholder_end..placeholder.token.start]);
            last_placeholder_end = placeholder.token.end;

            match param {
                BindParam::Scalar => {
                    if placeholder.kleene.is_some() {
                        return err!("expected vector bind param for {:?}", placeholder);
                    }

                    push_placeholder(&mut out);
                }
                BindParam::Vector(len) => {
                    let kleene = placeholder.kleene.ok_or_else(|| {
                        Error::Expand(format!("expected Kleene operator for {:?}", placeholder))
                    })?;

                    if len == 0 {
                        match kleene {
                            Kleene::Plus => {
                                out.push_str("NULL");
                            }
                        }
                        continue;
                    }

                    let mut comma_needed = false;

                    for _ in 0..len {
                        if comma_needed {
                            out.push_str(", ");
                        }

                        push_placeholder(&mut out);

                        comma_needed = true;
                    }
                }
            }
        }

        if !placeholders.as_slice().is_empty() {
            return err!("unused placeholders: {:?}", placeholders.as_slice());
        }

        let unused_params = params.count();

        if unused_params != 0 {
            return err!("{} unused BindParams", unused_params);
        }

        out.push_str(&self.query[last_placeholder_end..]);

        Ok(out.into())
    }
}

impl std::error::Error for Error {}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse { line, column, message, context, .. } => {
                write!(
                    f,
                    "Error parsing placeholders in query at line {}, column {}: {} near {:?}",
                    line, column, message, context
                )
            }
            Self::Expand(s) => write!(f, "Error expanding placeholders in query: {}", s),
        }
    }
}

impl From<Error> for crate::Error {
    fn from(e: Error) -> Self {
        crate::Error::Placeholders(e)
    }
}

/// similar to combine's `IndexPositioner` but which correctly maintains byte-position
/// and also tracks a `SourcePosition` for user-friendliness
#[derive(Clone, Default, PartialOrd, Ord, PartialEq, Eq, Debug)]
struct StrPosition {
    byte_pos: usize,
    source_pos: SourcePosition,
}

impl Positioner<char> for StrPosition {
    type Position = Self;
    type Checkpoint = Self;

    fn position(&self) -> Self::Position {
        self.clone()
    }

    fn update(&mut self, token: &char) {
        self.byte_pos += token.len_utf8();
        self.source_pos.update(token);
    }

    fn checkpoint(&self) -> Self::Checkpoint {
        self.clone()
    }

    fn reset(&mut self, checkpoint: Self::Checkpoint) {
        *self = checkpoint;
    }
}

impl<'a> RangePositioner<char, &'a str> for StrPosition {
    fn update_range(&mut self, range: &&'a str) {
        self.byte_pos += range.len();
        self.source_pos.update_range(range);
    }
}

impl Display for StrPosition {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.source_pos.fmt(f)
    }
}

struct DisplayErrors<'a>(Vec<combine::easy::Error<char, &'a str>>);

impl Display for DisplayErrors<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        combine::easy::Error::fmt_errors(&self.0, f)
    }
}

pub fn parse_query(query: &str) -> Result<ParsedQuery<'_>> {
    let placeholders = parse_query_string(query).map_err(|e| {
        let combine::easy::Errors {
            position: StrPosition { byte_pos, source_pos: SourcePosition { line, column } },
            errors,
        } = e;

        Error::Parse {
            byte_position: byte_pos,
            line,
            column,
            message: DisplayErrors(errors).to_string(),
            context: error_context(query, byte_pos).to_string(),
        }
    })?;

    Ok(ParsedQuery { query, placeholders })
}

fn parse_query_string(
    query: &str,
) -> Result<Vec<Placeholder<'_>>, combine::easy::Errors<char, &'_ str, StrPosition>> {
    parse_placeholders()
        .easy_parse(combine::stream::position::Stream::with_positioner(
            query,
            StrPosition::default(),
        ))
        .map(|(placeholders, _)| placeholders)
}

fn parse_placeholders<'a, I: RangeStream<Token = char, Range = &'a str, Position = StrPosition>>(
) -> impl Parser<combine::easy::Stream<I>, Output = Vec<Placeholder<'a>>> {
    combine::many(
        repeat_skip_until(
            combine::choice((one_of("'\"`".chars()).then(escaped_string), any().map(|_| ()))),
            attempt(token('{')),
        )
        .then(|_| parse_placeholder()),
    )
}

fn parse_placeholder<'a, I: RangeStream<Token = char, Range = &'a str, Position = StrPosition>>(
) -> impl Parser<I, Output = Placeholder<'a>> {
    (
        position(),
        recognize_with_value(between(
            token('{'),
            token('}'),
            (parse_ident(), optional(parse_kleene())),
        )),
    )
        .map(
            |(pos, (range, (ident, kleene))): (
                StrPosition,
                (&str, (Ident<'_>, Option<Kleene>)),
            )| {
                let pos = pos.byte_pos;
                Placeholder { token: pos..pos + range.len(), ident, kleene }
            },
        )
}

fn parse_ident<'a, I: RangeStream<Token = char, Range = &'a str>>(
) -> impl Parser<I, Output = Ident<'a>> {
    let ident = || (letter().or(token('_')), skip_many(alpha_num().or(token('_'))));

    choice((
        // explicit positional: `{N...}`
        parse_u16().map(Ident::Positional),
        // explicit identifier: `{foo...}`
        recognize(ident()).map(|ident: &str| Ident::Named(ident.into())),
        // field access: `{(foo.bar)...}`
        between(
            token('('),
            token(')'),
            recognize((skip_many(attempt((ident(), token('.')))), ident())),
        )
        .map(|ident: &str| Ident::Field(ident.into())),
        // implicit: `{...}`
        attempt(optional(parse_kleene())).map(|_| Ident::Implicit),
    ))
}

fn parse_kleene<I: Stream<Token = char>>() -> impl Parser<I, Output = Kleene> {
    // if we decide to support more Kleene operators
    // choice((
    //     token('?').map(|_| Kleene::Question),
    //     token('*').map(|_| Kleene::Star),
    //     token('+').map(|_| Kleene::Plus),
    // ))

    not_followed_by(choice((token('?'), token('*'))))
        .message("unsupported Kleene operator")
        .then(|_| token('+').map(|_| Kleene::Plus))
}

fn parse_u16<'a, I: RangeStream<Token = char, Range = &'a str>>() -> impl Parser<I, Output = u16> {
    from_str(take_while1(|c: char| c.is_digit(10)))
}

fn escaped_string<I: RangeStream<Token = char>>(quote_char: char) -> impl Parser<I, Output = ()>
where
    I::Range: combine::stream::Range,
{
    (
        escaped(take_while1(move |c| c != quote_char && c != '\\'), '\\', token(quote_char)),
        token(quote_char),
    )
        .map(|_| ())
}

/// Give context for the error in `s` at `at`
fn error_context(s: &str, at: usize) -> &str {
    // break the context at whitespace or non-printable character
    let break_context = |c: char| c.is_ascii_control() || c.is_whitespace();

    // get the byte position _after_ the first matched character before the given position
    // note that we don't want to break in the middle of a multibyte character, however unlikely
    // defaults to the beginning of the string
    let start = s[..at].rmatch_indices(break_context).next().map_or(0, |(i, sp)| i + sp.len());
    // let the end of our string be the byte position of the _next_ matching character or the length
    let end = s[at..].find(break_context).map_or(s.len(), |i| at + i);

    &s[start..end]
}

#[test]
fn test_parse_query_string() -> Result<(), Box<dyn std::error::Error>> {
    use Ident::*;
    use Kleene::*;

    assert_eq!(
        parse_query_string("SELECT 1 FROM foo WHERE bar = {} AND baz = {baz}")?,
        [
            Placeholder { token: 30..32, ident: Implicit, kleene: None },
            Placeholder { token: 43..48, ident: Named("baz".into()), kleene: None }
        ]
    );

    assert_eq!(
        parse_query_string("SELECT 1 FROM foo WHERE bar IN {(foo.bar)+}")?,
        [Placeholder { token: 31..43, ident: Field("foo.bar".into()), kleene: Some(Plus) }]
    );

    assert_eq!(
        parse_query_string(
            r#"SELECT 1 FROM foo WHERE quux = '{ "foo": "\'bar\'" }' and bar IN {0}"#
        )?,
        [Placeholder { token: 65..68, ident: Positional(0), kleene: None }]
    );

    Ok(())
}

#[test]
fn test_expand_parsed_query() -> Result<()> {
    use BindParam::*;
    use ParamIndexing::*;

    // Postgres
    assert_eq!(
        parse_query("SELECT 1 FROM foo WHERE bar = {} AND baz = {baz}")?.expand_inner(
            '$',
            OneIndexed,
            vec![Scalar; 2]
        )?,
        "SELECT 1 FROM foo WHERE bar = $1 AND baz = $2"
    );

    assert_eq!(
        parse_query(
            r#"
                SELECT 1 
                FROM foo 
                WHERE bar IN ({(foo.bar)+})
                AND baz IN ({baz+})
                AND quux IN ({quux+})"#
        )?
        .expand_inner('$', OneIndexed, vec![Vector(3), Vector(0), Vector(1)])?,
        r#"
                SELECT 1 
                FROM foo 
                WHERE bar IN ($1, $2, $3)
                AND baz IN (NULL)
                AND quux IN ($4)"#
    );

    assert_eq!(
        parse_query(r#"SELECT 1 FROM foo WHERE quux = '{ "foo": "\'bar\'" }' and bar IN {0}"#)?
            .expand_inner('$', OneIndexed, vec![Scalar])?,
        r#"SELECT 1 FROM foo WHERE quux = '{ "foo": "\'bar\'" }' and bar IN $1"#
    );

    // MySQL
    assert_eq!(
        parse_query("SELECT 1 FROM foo WHERE bar = {} AND baz = {baz}")?.expand_inner(
            '?',
            Implicit,
            vec![Scalar; 2]
        )?,
        "SELECT 1 FROM foo WHERE bar = ? AND baz = ?"
    );

    assert_eq!(
        parse_query(
            r#"
                SELECT 1 
                FROM foo 
                WHERE bar IN ({(foo.bar)+})
                AND baz IN ({baz+})
                AND quux IN ({quux+})"#
        )?
        .expand_inner('?', Implicit, vec![Vector(3), Vector(0), Vector(1)])?,
        r#"
                SELECT 1 
                FROM foo 
                WHERE bar IN (?, ?, ?)
                AND baz IN (NULL)
                AND quux IN (?)"#
    );

    assert_eq!(
        parse_query(r#"SELECT 1 FROM foo WHERE quux = '{ "foo": "\'bar\'" }' and bar IN {0}"#)?
            .expand_inner('?', Implicit, vec![Scalar])?,
        r#"SELECT 1 FROM foo WHERE quux = '{ "foo": "\'bar\'" }' and bar IN ?"#
    );

    Ok(())
}
