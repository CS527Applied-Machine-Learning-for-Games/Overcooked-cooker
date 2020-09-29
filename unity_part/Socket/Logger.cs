using System;
using System.IO;
using UnityEngine;

namespace Overcooked_Socket
{
    internal static class Logger
    {

        private readonly static string PATH = "";

        public static void Log(String message)
        {
            String str = File.ReadAllText(PATH);
            File.WriteAllText(PATH, str + message + "\n");
        }

        public static String FormatPosition(Vector3 location)
        {
            return "x=" + Math.Round(location.x, 2) + ", y=" + Math.Round(location.y, 2) + ", z=" + Math.Round(location.z, 2);
        }

        public static void Clear()
        {
            File.WriteAllText(PATH, "");
        }

    }
}
