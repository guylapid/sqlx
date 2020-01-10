use byteorder::LittleEndian;

use crate::decode::{Decode, DecodeError};
use crate::encode::Encode;
use crate::io::{Buf, BufMut};
use crate::mysql::protocol::TypeId;
use crate::mysql::types::MySqlTypeInfo;
use crate::mysql::MySql;
use crate::types::HasSqlType;

impl HasSqlType<i8> for MySql {
    fn type_info() -> MySqlTypeInfo {
        MySqlTypeInfo::new(TypeId::TINY_INT)
    }
}

impl Encode<MySql> for i8 {
    fn encode(&self, buf: &mut Vec<u8>) {
        buf.push(*self as u8);
    }
}

impl Decode<MySql> for i8 {
    fn decode(buf: &[u8]) -> Result<Self, DecodeError> {
        Ok(buf[0] as i8)
    }
}

impl HasSqlType<i16> for MySql {
    fn type_info() -> MySqlTypeInfo {
        MySqlTypeInfo::new(TypeId::SMALL_INT)
    }
}

impl Encode<MySql> for i16 {
    fn encode(&self, buf: &mut Vec<u8>) {
        buf.put_i16::<LittleEndian>(*self);
    }
}

impl Decode<MySql> for i16 {
    fn decode(mut buf: &[u8]) -> Result<Self, DecodeError> {
        buf.get_i16::<LittleEndian>().map_err(Into::into)
    }
}

impl HasSqlType<i32> for MySql {
    fn type_info() -> MySqlTypeInfo {
        MySqlTypeInfo::new(TypeId::INT)
    }
}

impl Encode<MySql> for i32 {
    fn encode(&self, buf: &mut Vec<u8>) {
        buf.put_i32::<LittleEndian>(*self);
    }
}

impl Decode<MySql> for i32 {
    fn decode(mut buf: &[u8]) -> Result<Self, DecodeError> {
        buf.get_i32::<LittleEndian>().map_err(Into::into)
    }
}

impl HasSqlType<i64> for MySql {
    fn type_info() -> MySqlTypeInfo {
        MySqlTypeInfo::new(TypeId::BIG_INT)
    }
}

impl Encode<MySql> for i64 {
    fn encode(&self, buf: &mut Vec<u8>) {
        buf.put_u64::<LittleEndian>(*self as u64);
    }
}

impl Decode<MySql> for i64 {
    fn decode(mut buf: &[u8]) -> Result<Self, DecodeError> {
        buf.get_u64::<LittleEndian>()
            .map_err(Into::into)
            .map(|val| val as i64)
    }
}
