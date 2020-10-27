using System;
using System.Collections;
using UnityEngine;
using Object = UnityEngine.Object;
namespace Overcooked_Socket
{
    internal static class ContainerUtil
    {
        public static Vector3 GetContainerPosition(ServerIngredientContainer container)
        {
            return container.transform.position;

        }
        public static bool HasIngredient(ServerIngredientContainer container)
        {
            return GetIngredient(container) != "";
        }
        public static String GetIngredient(ServerIngredientContainer container)
        {
            String result = "";
            Boolean flag = true;
            AssembledDefinitionNode[] containerContents = container.GetContents();
            ArrayList contents = new ArrayList();
            foreach (AssembledDefinitionNode node in containerContents)
            {
                ItemUtil.GetIngredientsInNode(node, contents);
            }
            foreach (String content in contents)
            {
                if(flag == true)
                {
                    result = result + content;
                    flag = false;
                }
                else
                {
                    result = result + "+" + content;
                }
            }
            return result;
        }
    }
}
