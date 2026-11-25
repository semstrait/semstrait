use substrait::proto::extensions::simple_extension_declaration::{ExtensionFunction, MappingType};
use substrait::proto::extensions::{SimpleExtensionDeclaration, SimpleExtensionUrn};

use crate::error::{number, InvalidRewrite};
use crate::functions::FunctionName;

#[derive(Debug, Default)]
pub struct Extensions {
    urns: Vec<&'static str>,
    functions: Vec<(u32, &'static str)>,
}

impl Extensions {
    pub fn declare(&mut self, function: FunctionName) -> Result<u32, InvalidRewrite> {
        let urn = anchor(&mut self.urns, function.urn)?;
        anchor(&mut self.functions, (urn, function.name))
    }

    pub(crate) fn declarations(self) -> (Vec<SimpleExtensionUrn>, Vec<SimpleExtensionDeclaration>) {
        let urns = (self.urns.into_iter().zip(1..))
            .map(|(urn, extension_urn_anchor)| SimpleExtensionUrn {
                extension_urn_anchor,
                urn: urn.to_owned(),
            })
            .collect();
        let functions = (self.functions.into_iter().zip(1..))
            .map(
                |((extension_urn_reference, name), function_anchor)| SimpleExtensionDeclaration {
                    mapping_type: Some(MappingType::ExtensionFunction(ExtensionFunction {
                        extension_urn_reference,
                        function_anchor,
                        name: name.to_owned(),
                    })),
                },
            )
            .collect();
        (urns, functions)
    }
}

fn anchor<T: PartialEq>(items: &mut Vec<T>, item: T) -> Result<u32, InvalidRewrite> {
    let position = match items.iter().position(|known| *known == item) {
        Some(position) => position,
        None => {
            items.push(item);
            items.len() - 1
        }
    };
    number(position + 1)
}
